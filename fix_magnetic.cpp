/* ----------------------------------------------------------------------
    This is the

    ██╗     ██╗ ██████╗  ██████╗  ██████╗ ██╗  ██╗████████╗███████╗
    ██║     ██║██╔════╝ ██╔════╝ ██╔════╝ ██║  ██║╚══██╔══╝██╔════╝
    ██║     ██║██║  ███╗██║  ███╗██║  ███╗███████║   ██║   ███████╗
    ██║     ██║██║   ██║██║   ██║██║   ██║██╔══██║   ██║   ╚════██║
    ███████╗██║╚██████╔╝╚██████╔╝╚██████╔╝██║  ██║   ██║   ███████║
    ╚══════╝╚═╝ ╚═════╝  ╚═════╝  ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚══════╝®

    DEM simulation engine, released by
    DCS Computing Gmbh, Linz, Austria
    http://www.dcs-computing.com, office@dcs-computing.com

    LIGGGHTS® is part of CFDEM®project:
    http://www.liggghts.com | http://www.cfdem.com

    Core developer and main author:
    Christoph Kloss, christoph.kloss@dcs-computing.com

    LIGGGHTS® is open-source, distributed under the terms of the GNU Public
    License, version 2 or later. It is distributed in the hope that it will
    be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
    of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. You should have
    received a copy of the GNU General Public License along with LIGGGHTS®.
    If not, see http://www.gnu.org/licenses . See also top-level README
    and LICENSE files.

    LIGGGHTS® and CFDEM® are registered trade marks of DCS Computing GmbH,
    the producer of the LIGGGHTS® software and the CFDEM®coupling software
    See http://www.cfdem.com/terms-trademark-policy for details.

-------------------------------------------------------------------------
    Contributing author and copyright for this file:
    Thomas Leps
    University of Maryland College Park
    tjleps@gmail.com
------------------------------------------------------------------------- */
#include "fix_magnetic.h"
#include <eigen-3.4.0/Eigen/Dense>
#include <iostream>
#include "spherical_harmonics.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <iostream>
#include "atom.h"
#include "update.h"
#include "modify.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "force.h"
#include "respa.h"
#include "input.h"
#include "variable.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;
using namespace FixConst;

enum{CONSTANT,EQUAL,ATOM};

/* ---------------------------------------------------------------------- */

FixMagnetic::FixMagnetic(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg != 6) error->all(FLERR,"Illegal fix magnetic command");

  xstr = ystr = zstr = NULL;

  if (strstr(arg[3],"v_") == arg[3]) {
    int n = strlen(&arg[3][2]) + 1;
    xstr = new char[n];
    strcpy(xstr,&arg[3][2]);
  } else {
    ex = atof(arg[3]);
    xstyle = CONSTANT;
  }

  if (strstr(arg[4],"v_") == arg[4]) {
    int n = strlen(&arg[4][2]) + 1;
    ystr = new char[n];
    strcpy(ystr,&arg[4][2]);
  } else {
    ey = atof(arg[4]);
    ystyle = CONSTANT;
  }

  if (strstr(arg[5],"v_") == arg[5]) {
    int n = strlen(&arg[5][2]) + 1;
    zstr = new char[n];
    strcpy(zstr,&arg[5][2]);
  } else {
    ez = atof(arg[5]);
    zstyle = CONSTANT;
  }

  maxatom = 0;
  hfield = NULL;
}

/* ---------------------------------------------------------------------- */

FixMagnetic::~FixMagnetic()
{
  delete [] xstr;
  delete [] ystr;
  delete [] zstr;
  memory->destroy(hfield);

  if (susceptibility_)
    delete []susceptibility_;
}

/* ---------------------------------------------------------------------- */

int FixMagnetic::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  mask |= POST_FORCE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixMagnetic::init()
{
  //if (!atom->q_flag) error->all(FLERR,"Fix hfield requires atom attribute q");

  // check variables

  if (xstr) {
    xvar = input->variable->find(xstr);
    if (xvar < 0) error->all(FLERR,
                             "Variable name for fix hfield does not exist");
    if (input->variable->equalstyle(xvar)) xstyle = EQUAL;
    else if (input->variable->atomstyle(xvar)) xstyle = ATOM;
    else error->all(FLERR,"Variable for fix hfield is invalid style");
  }
  if (ystr) {
    yvar = input->variable->find(ystr);
    if (yvar < 0) error->all(FLERR,
                             "Variable name for fix hfield does not exist");
    if (input->variable->equalstyle(yvar)) ystyle = EQUAL;
    else if (input->variable->atomstyle(yvar)) ystyle = ATOM;
    else error->all(FLERR,"Variable for fix hfield is invalid style");
  }
  if (zstr) {
    zvar = input->variable->find(zstr);
    if (zvar < 0) error->all(FLERR,
                             "Variable name for fix hfield does not exist");
    if (input->variable->equalstyle(zvar)) zstyle = EQUAL;
    else if (input->variable->atomstyle(zvar)) zstyle = ATOM;
    else error->all(FLERR,"Variable for fix hfield is invalid style");
  }

  if (xstyle == ATOM || ystyle == ATOM || zstyle == ATOM)
    varflag = ATOM;
  else if (xstyle == EQUAL || ystyle == EQUAL || zstyle == EQUAL)
    varflag = EQUAL;
  else varflag = CONSTANT;

  //Neighbor List

  int irequest = neighbor->request((void *) this);
  neighbor->requests[irequest]->pair = 0;
  neighbor->requests[irequest]->fix = 1;
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;

  if (strstr(update->integrate_style,"respa"))
    nlevels_respa = ((Respa *) update->integrate)->nlevels;

  // error checks on coarsegraining
  if(force->cg_active())
    error->cg(FLERR,this->style);

  int max_type = atom->get_properties()->max_type();

  if (susceptibility_) delete []susceptibility_;
  susceptibility_ = new double[max_type];
  fix_susceptibility_ =
    static_cast<FixPropertyGlobal*>(modify->find_fix_property("magneticSusceptibility","property/global","peratomtype",max_type,0,style));

  // pre-calculate susceptibility for possible contact material combinations
  for(int i=1;i< max_type+1; i++)
      for(int j=1;j<max_type+1;j++)
      {
          susceptibility_[i-1] = fix_susceptibility_->compute_vector(i-1);
          if(susceptibility_[i-1] <= 0.)
            error->all(FLERR,"Fix magnetic: magnetic susceptibility must not be <= 0");
      }
}

/* ---------------------------------------------------------------------- */

void FixMagnetic::init_list(int id, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void FixMagnetic::setup(int vflag)
{
  if (strstr(update->integrate_style,"verlet"))
    post_force(vflag);
  else {
    ((Respa *) update->integrate)->copy_flevel_f(nlevels_respa-1);
    post_force_respa(vflag,nlevels_respa-1,0);
    ((Respa *) update->integrate)->copy_f_flevel(nlevels_respa-1);
  }
}

/* ----------------------------------------------------------------------*/


void FixMagnetic::post_force(int vflag)
{
  //std::cout<<"made it this far"<<std::endl;
  int i,j,ii,jj,inum,jnum;
  int *ilist,*jlist,*numneigh,**firstneigh,*slist;
  double sep_sq, sep, mir, mjr, mumu, A, K, muR, susc, susc_eff;
  double *rad = atom->radius;
  double **x = atom->x;
  double **mu = atom->mu;
  double **f = atom->f;
  double *q = atom->q;
  double p4 = M_PI*4;
  double mu0 = p4*1e-7; //for SI units changed by Anmol
  //double u=1;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int *type = atom->type;
  // reallocate hfield array if necessary
  Eigen::Vector3d SEP;

  if (varflag == ATOM && nlocal > maxatom) {
    maxatom = atom->nmax;
    memory->destroy(hfield);
    memory->create(hfield,maxatom,3,"hfield:hfield");
  }
  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  std::cout<<inum<<std::endl;
  
  if (varflag == CONSTANT) {
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];

      if (mask[i] & groupbit) {
        susc= susceptibility_[type[i]-1];
        susc_eff=3*susc/(susc+3);//3*(susc-1)/(susc+2);
        double C = susc_eff*rad[i]*rad[i]*rad[i]*p4/3/mu0;
        mu[i][0] = C*ex;
        mu[i][1] = C*ey;
        mu[i][2] = C*ez;
        jlist = firstneigh[i];
        jnum = numneigh[i];
        Eigen::Vector3d mu_i_vector;
        mu_i_vector << mu[i][0], mu[i][1], mu[i][2];

        for (jj = 0; jj<jnum; jj++)  {          
          j =jlist[jj];
          j &= NEIGHMASK;
          SEP << x[i][0] - x[j][0], x[i][1] - x[j][1], x[i][2] - x[j][2];
          sep = SEP.norm();
          // r = sqrt(rsq);
          A = C*mu0/p4/sep/sep/sep;
          // SEP /= sep;
          
          Eigen::Vector3d mu_j_vector;
          mu_j_vector << mu[j][0], mu[j][1], mu[j][2];

          mjr = mu_j_vector.dot(SEP)/sep;

          mu_i_vector += A*(3*mjr*SEP/sep-mu_j_vector);
        }
        mu[i][0]=mu_i_vector[0];
        mu[i][1]=mu_i_vector[1];
        mu[i][2]=mu_i_vector[2];

        mumu = mu_i_vector.dot(mu_i_vector);

        if (mumu > C*C*4){
          muR=sqrt(C*C*4/mumu);
          mu[i][0]=mu_i_vector[0]*muR;
          mu[i][1]=mu_i_vector[1]*muR;
          mu[i][2]=mu_i_vector[2]*muR;
        }
      }
    }

    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];

      if (mask[i] & groupbit) {
        jlist = firstneigh[i];
        jnum = numneigh[i];
        Eigen::Vector3d mu_i_vector;
        mu_i_vector << mu[i][0], mu[i][1], mu[i][2];

        for (jj = 0; jj<jnum; jj++)  {
          j =jlist[jj];
          j &= NEIGHMASK;
          Eigen::Vector3d mu_j_vector;
          mu_j_vector << mu[j][0], mu[j][1], mu[j][2];
          
          SEP << x[i][0] - x[j][0], x[i][1] - x[j][1], x[i][2] - x[j][2];
          sep = SEP.norm();
          sep_sq = sep*sep;
          
          //K = 3e-4/rsq/rsq;
          K = 3e-7/sep_sq/sep_sq;
  
          // dx /= r;

          mir=mu_i_vector.dot(SEP)/sep;
          mjr=mu_j_vector.dot(SEP)/sep;
          mumu = mu_i_vector.dot(mu_j_vector);

          f[i][0] += K*(mir*mu[j][0]+mjr*mu[i][0]+(mumu-5*mjr*mir)*SEP[0]/sep);
          f[i][1] += K*(mir*mu[j][1]+mjr*mu[i][1]+(mumu-5*mjr*mir)*SEP[1]/sep);
          f[i][2] += K*(mir*mu[j][2]+mjr*mu[i][2]+(mumu-5*mjr*mir)*SEP[2]/sep);
            
        }
      }
    }
    // Spherical Harmonics
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];

      if (mask[i] & groupbit) {
        susc= susceptibility_[type[i]-1];
        susc_eff=3*susc/(susc+3);
        double C = susc_eff*rad[i]*rad[i]*rad[i]*p4/3/mu0;
        Eigen::Vector3d mu_i_dipole;
        mu_i_dipole<<C*ex, C*ey, C*ez;
        jlist = firstneigh[i];
        jnum = numneigh[i];
        Eigen::Vector3d mu_i_vector;
        mu_i_vector << mu[i][0], mu[i][1], mu[i][2];
        std::cout<<"jlist"<<jnum<<std::endl<<std::endl;

        for (jj = 0; jj<jnum; jj++)  {
          j =jlist[jj];
          j &= NEIGHMASK;
          Eigen::Vector3d mu_j_vector;
          std::cout<<"i: "<<i<<"j: "<<j<<std::endl<<std::endl;
          mu_j_vector << mu[j][0], mu[j][1], mu[j][2];
          
          SEP << x[i][0] - x[j][0], x[i][1] - x[j][1], x[i][2] - x[j][2];
          sep = SEP.norm();
          // r = sqrt(rsq);
          A = C*mu0/p4/sep/sep/sep;
          double mr = mu_i_dipole.dot(SEP)/sep;
          K = 3e-7/sep_sq/sep_sq;

          double mumu_d=mu_i_dipole.dot(mu_i_dipole);

          if (sep/rad[i] < 4.2){
            Eigen::Vector3d H0;
            H0<<ex, ey, ez;
            
            spherical_harmonics particle_i_j(rad[i], susc, H0, SEP, mu_i_vector);
            
            Eigen::Vector3d F_2B;
            F_2B=particle_i_j.get_force_actual_coord();
            // if (i==0){
            //   F_2B<<0,0,-1.32303550148865e-13;//particle_i_j.get_force();
            // }
            // else if (i==1)
            // {
            //   F_2B<<0,0,1.32303550148865e-13;//particle_i_j.get_force();
            // }
            

            f[i][0] += F_2B[0] - K*(mr*mu_i_dipole[0]+mr*mu_i_dipole[0]+(mumu_d-5*mr*mr)*SEP[0]/sep);
            f[i][1] += F_2B[1] - K*(mr*mu_i_dipole[1]+mr*mu_i_dipole[1]+(mumu_d-5*mr*mr)*SEP[1]/sep);
            f[i][2] += F_2B[2] - K*(mr*mu_i_dipole[2]+mr*mu_i_dipole[2]+(mumu_d-5*mr*mr)*SEP[2]/sep);

          }
        }
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixMagnetic::post_force_respa(int vflag, int ilevel, int iloop)
{
  if (ilevel == nlevels_respa-1) post_force(vflag);
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double FixMagnetic::memory_usage()
{
  double bytes = 0.0;
  if (varflag == ATOM) bytes = atom->nmax*3 * sizeof(double);
  return bytes;
}

double FixMagnetic::nchoosek(int n, int k){
    if (k > n) return 0;
    if (k * 2 > n) k = n-k;
    if (k == 0) return 1;

    int result = n;
    for( int i = 2; i <= k; ++i ) {
        result *= (n-i+1);
        result /= i;
    }
    return result;
}