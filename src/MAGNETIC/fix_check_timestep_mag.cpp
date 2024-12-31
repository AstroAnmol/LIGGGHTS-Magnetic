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
------------------------------------------------------------------------- */

#include <string.h>
#include <stdlib.h>
#include <cmath>
#include <algorithm>
#include "atom.h"
#include "update.h"
#include "error.h"
#include "fix_check_timestep_mag.h"
#include "fix_check_timestep_gran.h"
#include "pair_gran.h"
#include "properties.h"
#include "fix_property_global.h"
#include "force.h"
#include "comm.h"
#include "modify.h"
#include "fix_wall_gran.h"
// #include "fix_mesh_surface.h"
#include "fix_magnetic.h"
#include "neighbor.h"
#include "mpi_liggghts.h"
#include "property_registry.h"
#include "global_properties.h"

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MODEL_PARAMS;

/* ---------------------------------------------------------------------- */

FixCheckTimestepMag::FixCheckTimestepMag(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  warnflag = true;
  errorflag = false;
  vmax_user = 0.;

  if (narg < 5) error->all(FLERR,"Illegal fix check/timestep/mag command, not enough arguments");

  int iarg = 5;

  if(0 == strcmp("check_every_time",arg[3]))
  {
      nevery = atoi(arg[4]);
      dF_by_F = atof(arg[5]);
      iarg = 6;
  }
  else
  {
      nevery = atoi(arg[3]);
      dF_by_F = atof(arg[4]);
      iarg = 5;
  }

  while(iarg < narg)
  {
      if (strcmp(arg[iarg],"warn") == 0) {
          if (narg < iarg+2) error->fix_error(FLERR,this,"not enough arguments for 'warn'");
          if(0 == strcmp(arg[iarg+1],"no"))
            warnflag = false;
          else if(0 == strcmp(arg[iarg+1],"yes"))
            warnflag = true;
          else
            error->fix_error(FLERR,this,"expecting 'yes' or 'no' after 'warn'");
          iarg += 2;
      } else if (strcmp(arg[iarg],"error") == 0) {
          if (narg < iarg+2) error->fix_error(FLERR,this,"not enough arguments for 'error'");
          if(0 == strcmp(arg[iarg+1],"no"))
            errorflag = false;
          else if(0 == strcmp(arg[iarg+1],"yes"))
            errorflag = true;
          else
            error->fix_error(FLERR,this,"expecting 'yes' or 'no' after 'error'");
          iarg += 2;
      } else if (strcmp(arg[iarg],"vmax") == 0) {
          if (narg < iarg+2) error->fix_error(FLERR,this,"not enough arguments for 'vmax'");
          vmax_user = force->numeric(FLERR,arg[iarg+1]);
          iarg += 2;
      }  else if(strcmp(style,"mesh/surface") == 0) {
          char *errmsg = new char[strlen(arg[iarg])+50];
          sprintf(errmsg,"unknown keyword or wrong keyword order: %s", arg[iarg]);
          error->fix_error(FLERR,this,errmsg);
          delete []errmsg;
      }
  }

  vector_flag = 1;
  size_vector = 3;
  global_freq = nevery;
  extvector = 1;
  
  fraction_magnetic = 0.;
}

/* ---------------------------------------------------------------------- */

FixCheckTimestepMag::~FixCheckTimestepMag(){
  if (susceptibility_)
    delete []susceptibility_;
}


/* ---------------------------------------------------------------------- */

int FixCheckTimestepMag::setmask()
{
  int mask = 0;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixCheckTimestepMag::init()
{
  //some error checks
  if(!atom->radius_flag || !atom->density_flag)
    error->all(FLERR,"Fix check/timestep/mag can only be used together with atom style sphere");

  pg = (PairGran*)force->pair_match("gran",1);
  if(!pg) pg = (PairGran*)force->pair_match("gran/omp",1);

  if (!pg)
    error->all(FLERR,"Fix check/timestep/mag can only be used together with: gran");

  Fix *fix_magnetic = modify->find_fix_style("magnetic",0); 
  if (fix_magnetic == NULL) {
    error->all(FLERR, "Fix check/timestep/mag can only be used together with: magnetic");
  }

  Fix *fix_check_timestep_gran = modify->find_fix_style("check/timestep/gran",0); 
  if (fix_check_timestep_gran == NULL) {
    error->all(FLERR, "Fix check/timestep/mag can only be used together with: check/timestep/gran");
  }

  properties = atom->get_properties();
  int max_type = properties->max_type();
  
  if (susceptibility_) delete []susceptibility_;
  fix_susceptibility_ = static_cast<FixPropertyGlobal*>(modify->find_fix_property("magneticSusceptibility","property/global","peratomtype",max_type,0,style));
  susceptibility_ = new double[max_type];
  
  for(int i=1;i< max_type+1; i++){
    susceptibility_[i-1] = fix_susceptibility_->compute_vector(i-1);
    if(susceptibility_[i-1] <= 0.)
      error->all(FLERR,"Fix magnetic: magnetic susceptibility must not be <= 0");
  }

  if(!fix_susceptibility_)
    error->all(FLERR,"Fix check/timestep/mag only works with a magnetic susceptibility");


  FixMagnetic *fm = (FixMagnetic *) fix_magnetic; 
  mag_timestep_value = fm->N_magforce_timestep; 

  fwg = NULL;
  for (int i = 0; i < modify->n_fixes_style("wall/gran"); i++)
      if(static_cast<FixWallGran*>(modify->find_fix_style("wall/gran",i))->is_mesh_wall())
        fwg = static_cast<FixWallGran*>(modify->find_fix_style("wall/gran",i));


  // FixCheckTimestepGran *fctg = (FixCheckTimestepGran *) fix_check_timestep_gran;
  // v_rel_max_sim = fctg->v_rel_max_simulation;

}

/* ---------------------------------------------------------------------- */

void FixCheckTimestepMag::end_of_step()
{
    calc_magnetic_estims();

    double dt = update->dt;
    fraction_magnetic = dt*mag_timestep_value/magnetic_time;

    fraction_magnetic_lim = 0.8;

    if(errorflag || (warnflag && comm->me==0))
    {
        char errstr[512];

        if(fraction_magnetic > fraction_magnetic_lim)
        {   
            sprintf(errstr,"time-step is %f %% of magnetic time",fraction_magnetic*100.);
            if(errorflag)
                error->fix_error(FLERR,this,errstr);
            else
                error->warning(FLERR,errstr);
        }
    }
}

/* ---------------------------------------------------------------------- */

void FixCheckTimestepMag::calc_magnetic_estims()
{
  int *type = atom->type;
  double **v = atom->v;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

   //check rayleigh time and vmax of particles
  // rayleigh_time = BIG;
  // r_min = BIG;
  double vmax_sqr = 0;
  double vmag_sqr;
  // double rayleigh_time_i;

  for (int i = 0; i < nlocal; i++)
  {
    if (mask[i] & groupbit)
    {
        // double rad = r[i];
        // #ifdef SUPERQUADRIC_ACTIVE_FLAG
        // if(atom->superquadric_flag)
        //     rad=std::min(std::min(atom->shape[i][0],atom->shape[i][1]),atom->shape[i][2]);        
        // #endif

        // double shear_mod = Y->get_values()[type[i]-1]/(2.*(nu->get_values()[type[i]-1]+1.));
        // rayleigh_time_i = M_PI*rad*sqrt(density[i]/shear_mod)/(0.1631*nu->get_values()[type[i]-1]+0.8766);
        // if(rayleigh_time_i < rayleigh_time) rayleigh_time = rayleigh_time_i;

        vmag_sqr = vectorMag3DSquared(v[i]);
        if(vmag_sqr > vmax_sqr)
            vmax_sqr = vmag_sqr;

        // if(rad < r_min) r_min = rad;
    }
  }

  // MPI_Min_Scalar(r_min,world);
  MPI_Max_Scalar(vmax_sqr,world);
  // MPI_Min_Scalar(rayleigh_time,world);

  //choose vmax_user if larger than value in simulation
  if(vmax_user*vmax_user > vmax_sqr)
    vmax_sqr = vmax_user*vmax_user;

  // get vmax of geometry
  FixMeshSurface ** mesh_list;
  TriMesh * mesh;
  double *v_node;
  double vmax_sqr_mesh=0.;
  double vmag_sqr_mesh;

  if(fwg)
  {
      mesh_list = fwg->mesh_list();
      for(int imesh = 0; imesh < fwg->n_meshes(); imesh++)
      {
          mesh = (mesh_list[imesh])->triMesh();
          if(mesh->isMoving())
          {
              // check if perElementProperty 'v' exists
              if (mesh->prop().getElementPropertyIndex("v") == -1)
                  error->one(FLERR,"Internal error - mesh has no perElementProperty 'v' \n");
              // loop local elements only
              int sizeMesh = mesh->sizeLocal();
              for(int itri = 0; itri < sizeMesh; itri++)
              {
                  for(int inode = 0; inode < 3; inode++)
                  {
                      v_node = mesh->prop().getElementProperty<MultiVectorContainer<double,3,3> >("v")->begin()[itri][inode];
                      vmag_sqr_mesh = vectorMag3DSquared(v_node);
                      if(vmag_sqr_mesh > vmax_sqr_mesh)
                        vmax_sqr_mesh = vmag_sqr_mesh;
                  }
              }
          }
      }
  }

  MPI_Max_Scalar(vmax_sqr_mesh,world);

  // decide vmax - either particle-particle or particle-mesh contact
  v_rel_max_sim = std::max(2.*sqrt(vmax_sqr),sqrt(vmax_sqr) + sqrt(vmax_sqr_mesh));

  double mag_time_min = 100000.;
  double mag_time_i;

  if ( !MathExtraLiggghts::compDouble(v_rel_max_sim,0.0) ) {
    for(int i = 0; i < nlocal; i++){
        if (mask[i] & groupbit){
                    
            const double susc= susceptibility_[type[i]-1];

            mag_time_i = -dF_by_F*MagForceEmperical(susc)/d_MagForceEmperical(susc)/v_rel_max_sim;

            if(mag_time_i<mag_time_min)
                mag_time_min=mag_time_i;
        }
    }
  }

  MPI_Min_Scalar(mag_time_min,world);
  magnetic_time = mag_time_min;

}

// Function to calculate the magnetic force
double FixCheckTimestepMag::MagForceEmperical(double susc){
    
    double sep = 2;
    double effSusc = 3.0 * susc / (3.0 + susc);

    double a1, b1, c1, d1, a2, b2, c2, d2, a3, b3, c3, d3;
    a1 = -497.95224865; b1 = 2671.56697629; c1 = -5718.94770409; d1 = 4061.8828376;
    a2 = 2944.52379528; b2 = -16054.0668261; c2 = 33639.65154115; d2 = -23897.55494789;
    a3 = -4317.45216271; b3 = 23490.26719675; c3 = -48875.99378488; d3 = 34745.55102049;

    double polynomialTerm1 = a1 * std::pow(effSusc, 3) + b1 * std::pow(effSusc, 2) + c1 * effSusc + d1;
    double polynomialTerm2 = a2 * std::pow(effSusc, 3) + b2 * std::pow(effSusc, 2) + c2 * effSusc + d2;
    double polynomialTerm3 = a3 * std::pow(effSusc, 3) + b3 * std::pow(effSusc, 2) + c3 * effSusc + d3;

    double F_DM = 4.0 * M_PI * std::pow(effSusc, 2) / (3.0 * std::pow(sep, 4));

    double result = mu0 * (F_DM + polynomialTerm1 / std::pow(sep, 5) 
                                              + polynomialTerm2 / std::pow(sep, 6) 
                                              + polynomialTerm3 / std::pow(sep, 7));

    return result;
}

// Function to calculate the derivative of magnetic force
double FixCheckTimestepMag::d_MagForceEmperical(double susc){
    
    double sep = 2;
    double effSusc = 3.0 * susc / (3.0 + susc);

    double a1, b1, c1, d1, a2, b2, c2, d2, a3, b3, c3, d3;
    a1 = -497.95224865; b1 = 2671.56697629; c1 = -5718.94770409; d1 = 4061.8828376;
    a2 = 2944.52379528; b2 = -16054.0668261; c2 = 33639.65154115; d2 = -23897.55494789;
    a3 = -4317.45216271; b3 = 23490.26719675; c3 = -48875.99378488; d3 = 34745.55102049;

    double polynomialTerm1 = a1 * std::pow(effSusc, 3) + b1 * std::pow(effSusc, 2) + c1 * effSusc + d1;
    double polynomialTerm2 = a2 * std::pow(effSusc, 3) + b2 * std::pow(effSusc, 2) + c2 * effSusc + d2;
    double polynomialTerm3 = a3 * std::pow(effSusc, 3) + b3 * std::pow(effSusc, 2) + c3 * effSusc + d3;

    double F_DM = 4.0 * M_PI * std::pow(effSusc, 2) / (3.0 * std::pow(sep, 4));

    double result = -mu0 * (4*F_DM / sep + 5*polynomialTerm1 / std::pow(sep, 6) 
                                              + 6*polynomialTerm2 / std::pow(sep, 7) 
                                              + 7*polynomialTerm3 / std::pow(sep, 8));

    return result;
}