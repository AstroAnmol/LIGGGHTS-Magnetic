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
    Anmol Sikka
    University of Maryland College Park
    anmolsikka09@gmail.com

    Thanks for the contributions by Thomas Leps
------------------------------------------------------------------------- 
    

*/
#include "fix_magnetic.h"
#include <Eigen/Dense>
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

#define EIGEN_DONT_PARALLELIZE

using namespace LAMMPS_NS;
using namespace FixConst;

enum{CONSTANT,EQUAL,ATOM};

/* ---------------------------------------------------------------------- */

FixMagnetic::FixMagnetic(LAMMPS *lmp, int narg, char **arg) :

  fix_susceptibility_(0),
  susceptibility_(0),
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
  // if (!atom->q_flag) error->all(FLERR,"Fix hfield requires atom attribute q");

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

  // Neighbor List

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
  int i,j,k,ii,jj,kk,inum,jnum;
  int *ilist,*jlist,*numneigh,**firstneigh,*slist;
  double *rad = atom->radius;
  double **x = atom->x;
  double **mu = atom->mu;
  double **f = atom->f;
  double *q = atom->q;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int *type = atom->type;

  // Variables needed for MDM calculations
    double p4 = M_PI*4;
    double mu0 = p4*1e-7;
    // External magnetic field
    Eigen::Vector3d H0;
    H0<<ex,ey,ez;

  // reallocate hfield array if necessary
  if (varflag == ATOM && nlocal > maxatom) {
    maxatom = atom->nmax;
    memory->destroy(hfield);
    memory->create(hfield,maxatom,3,"hfield:hfield");
  }
  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;
  
  if (varflag == CONSTANT) {

    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];

      if (mask[i] & groupbit) {
        // get neighbor list for the ith particle
        jlist = firstneigh[i];
        jnum = numneigh[i];

        // define vector for moment of ith particle
        Eigen::Vector3d mu_i_vector;

        // define 3N x 3N matrix for moment calculation (N is number of neighbors + 1)
        Eigen::MatrixXd mom_mat(3*(jnum+1), 3*(jnum+1));
        mom_mat=Eigen::MatrixXd::Zero(3*(jnum+1), 3*(jnum+1));
        Eigen::VectorXd H_vec(3*(jnum+1));
        Eigen::VectorXd mom_vec(3*(jnum+1));

        // First three terms of H_vec
        H_vec.head(3)=H0;

        // get susceptibility of particle ith particle
        double susc_i= susceptibility_[type[i]-1];
        double susc_eff_i=3*susc_i/(susc_i+3); // effective susceptibility

        // coefficient for ith particle
        double C_i = susc_eff_i*rad[i]*rad[i]*rad[i]*p4/3;

        // Define diagonal part of the matrix
        mom_mat(0,0)=1/C_i;
        mom_mat(1,1)=1/C_i;
        mom_mat(2,2)=1/C_i;
        
        // loop over each neighbor to get the first row/column of the matrix
        for (jj = 0; jj<jnum; jj++){          
          j =jlist[jj];
          j &= NEIGHMASK;

          // get susceptibility of particle jth particle
          double susc_j= susceptibility_[type[j]-1];
          double susc_eff_j=3*susc_j/(susc_j+3); // effective susceptibility

          // coefficient for ith particle
          double C_j = susc_eff_j*rad[j]*rad[j]*rad[j]*p4/3;

          // Define H_vec part
          H_vec.segment((jj+1)*3,3)=H0;
          
          // Define diagonal part of the matrix
          mom_mat(3*(jj+1),3*(jj+1))=1/C_j;
          mom_mat(3*(jj+1)+1,3*(jj+1)+1)=1/C_j;
          mom_mat(3*(jj+1)+2,3*(jj+1)+2)=1/C_j;
        
          // separation distance vector
          Eigen::Vector3d SEP_ij;
          SEP_ij << x[i][0] - x[j][0], x[i][1] - x[j][1], x[i][2] - x[j][2];
          double sep_ij = SEP_ij.norm();
          
          ////////////////////////////////////////////////
          // CHECK IF THE SEPARATION DISTANCE BETWEEN THE TWO PARTICLES
          // IS LOWER THAN THE SUM OF RADII.
          // CHANGE IT TO SUM OF RADII IF TRUE.
          ////////////////////////////////////////////////
          
          // sum of radii of two particles
          double rad_sum_ij = rad[i] + rad[j];
          if (sep_ij/rad_sum_ij<1)
          {
            SEP_ij=(SEP_ij/sep_ij)*rad_sum_ij;
            sep_ij=rad_sum_ij;
          }
          double p4_sep_ij_pow5_div_3=(p4*std::pow(sep_ij,5))/3;
          double inv_p4_sep_ij_pow3=1/(p4*pow(sep_ij,3));
          // i-j 3 X 3 matrix definition
          Eigen::Matrix3d mom_mat_ij;
          mom_mat_ij<< (SEP_ij(0)*SEP_ij(0)/p4_sep_ij_pow5_div_3 - inv_p4_sep_ij_pow3), SEP_ij(0)*SEP_ij(1)/p4_sep_ij_pow5_div_3, SEP_ij(0)*SEP_ij(2)/p4_sep_ij_pow5_div_3,
                       SEP_ij(1)*SEP_ij(0)/p4_sep_ij_pow5_div_3, (SEP_ij(1)*SEP_ij(1)/p4_sep_ij_pow5_div_3 - inv_p4_sep_ij_pow3 ), SEP_ij(1)*SEP_ij(2)/p4_sep_ij_pow5_div_3,
                       SEP_ij(2)*SEP_ij(0)/p4_sep_ij_pow5_div_3, SEP_ij(2)*SEP_ij(1)/p4_sep_ij_pow5_div_3, (SEP_ij(2)*SEP_ij(2)/p4_sep_ij_pow5_div_3 - inv_p4_sep_ij_pow3);
        
          mom_mat.block(0,(jj+1)*3,3,3)=-mom_mat_ij;
          mom_mat.block((jj+1)*3,0,3,3)=-mom_mat_ij;

          // loop over remaining neighbors for other rows
          for (kk = jj+1; kk < jnum; kk++){
            k =jlist[kk];
            k &=NEIGHMASK;

            // separation distance vector
            Eigen::Vector3d SEP_jk;
            SEP_jk << x[j][0] - x[k][0], x[j][1] - x[k][1], x[j][2] - x[k][2];
            double sep_jk = SEP_jk.norm();

            ////////////////////////////////////////////////
            // CHECK IF THE SEPARATION DISTANCE BETWEEN THE TWO PARTICLES
            // IS LOWER THAN THE SUM OF RADII.
            // CHANGE IT TO SUM OF RADII IF TRUE.
            ////////////////////////////////////////////////
          
            // sum of radii of two particles
            double rad_sum_jk = rad[j] + rad[k];
            if (sep_jk/rad_sum_jk<1)
            {
                SEP_jk=(SEP_jk/sep_jk)*rad_sum_jk;
                sep_jk=rad_sum_jk;
            }
            double p4_sep_jk_pow5_div_3=(p4*std::pow(sep_jk,5))/3;
            double inv_p4_sep_jk_pow3=1/(p4*pow(sep_jk,3));
            // j-k 3 X 3 matrix definition
            Eigen::Matrix3d mom_mat_jk;
            mom_mat_jk<< (SEP_jk(0)*SEP_jk(0)/p4_sep_jk_pow5_div_3 - inv_p4_sep_jk_pow3 ), SEP_jk(0)*SEP_jk(1)/p4_sep_jk_pow5_div_3, SEP_jk(0)*SEP_jk(2)/p4_sep_jk_pow5_div_3,
                        SEP_jk(1)*SEP_jk(0)/p4_sep_jk_pow5_div_3, (SEP_jk(1)*SEP_jk(1)/p4_sep_jk_pow5_div_3 - inv_p4_sep_jk_pow3 ), SEP_jk(1)*SEP_jk(2)/p4_sep_jk_pow5_div_3,
                        SEP_jk(2)*SEP_jk(0)/p4_sep_jk_pow5_div_3, SEP_jk(2)*SEP_jk(1)/p4_sep_jk_pow5_div_3, (SEP_jk(2)*SEP_jk(2)/p4_sep_jk_pow5_div_3 - inv_p4_sep_jk_pow3 );
        
            mom_mat.block((jj+1)*3,(kk+1)*3,3,3)=-mom_mat_jk;
            mom_mat.block((kk+1)*3,(jj+1)*3,3,3)=-mom_mat_jk;
          }
        }

        //solving the linear system of equations
        mom_vec=mom_mat.householderQr ().solve(H_vec);

        mu_i_vector=mom_vec.head(3);

        mu[i][0]=mu_i_vector[0];
        mu[i][1]=mu_i_vector[1];
        mu[i][2]=mu_i_vector[2];
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

          Eigen::Vector3d SEP_ij;
          SEP_ij << x[i][0] - x[j][0], x[i][1] - x[j][1], x[i][2] - x[j][2];
          double sep_ij = SEP_ij.norm();

          ////////////////////////////////////////////////
          // CHECK IF THE SEPARATION DISTANCE BETWEEN THE TWO PARTICLES
          // IS LOWER THAN THE SUM OF RADII.
          // CHANGE IT TO SUM OF RADII IF TRUE.
          ////////////////////////////////////////////////
          double rad_sum_ij = rad[i] + rad[j];
          if (sep_ij/rad_sum_ij<1)
          {
            SEP_ij=(SEP_ij/sep_ij)*rad_sum_ij;
            sep_ij=rad_sum_ij;
          }

          double mir=mu_i_vector.dot(SEP_ij)/sep_ij;
          double mjr=mu_j_vector.dot(SEP_ij)/sep_ij;
          double mumu = mu_i_vector.dot(mu_j_vector);
          
          f[i][0] += (3*mu0/p4/std::pow(sep_ij,4))*(mir*mu[j][0]+mjr*mu[i][0]+(mumu-5*mjr*mir)*SEP_ij[0]/sep_ij);
          f[i][1] += (3*mu0/p4/std::pow(sep_ij,4))*(mir*mu[j][1]+mjr*mu[i][1]+(mumu-5*mjr*mir)*SEP_ij[1]/sep_ij);
          f[i][2] += (3*mu0/p4/std::pow(sep_ij,4))*(mir*mu[j][2]+mjr*mu[i][2]+(mumu-5*mjr*mir)*SEP_ij[2]/sep_ij);
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

// double FixMagnetic::nchoosek(int n, int k){
//     if (k > n) return 0;
//     if (k * 2 > n) k = n-k;
//     if (k == 0) return 1;

//     int result = n;
//     for( int i = 2; i <= k; ++i ) {
//         result *= (n-i+1);
//         result /= i;
//     }
//     return result;
// }