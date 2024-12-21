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
#include "spherical_harmonics.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
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

#include <iostream>
#include "comm.h"

#define EIGEN_DONT_PARALLELIZE

using namespace LAMMPS_NS;
using namespace FixConst;

enum{CONSTANT,EQUAL,ATOM};

/* ---------------------------------------------------------------------- */

FixMagnetic::FixMagnetic(LAMMPS *lmp, int narg, char **arg) :

  fix_susceptibility_(0),
  susceptibility_(0),
  Fix(lmp, narg, arg),
  N_magforce_timestep(0)
{
  if (narg != 9) error->all(FLERR,"Illegal fix magnetic command");

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

  N_magforce_timestep = atof(arg[6]);

  // Store the model type argument
  model_type = arg[7];
  
  // Store the moment calculation type
  moment_calc = arg[8];

  maxatom = 0;
  // hfield = NULL;
}

/* ---------------------------------------------------------------------- */

FixMagnetic::~FixMagnetic()
{
  delete [] xstr;
  delete [] ystr;
  delete [] zstr;
  // memory->destroy(hfield);

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

  if (model_type == "mdm" || model_type == "inclusion") {

    // Update the current simulation time
    bigint current_timestep = update->ntimestep;

    // variable declarations
    int ii, i, inum;
    int *ilist;
    double **f = atom->f;
    double **mag_f = atom->mag_f;
    int *mask = atom->mask;
    // local atoms on this proc
    inum = list->inum;
    ilist = list->ilist;

    /* ----------------------------------------------------------------------
      Check if new force needs to be calculated
      Running the magnetic force model at a different cadence
    ------------------------------------------------------------------------- */
    if (current_timestep % N_magforce_timestep != 0){
      // Apply stored forces
      for (ii = 0; ii < inum; ii++) {
        i = ilist[ii];
        if (mask[i] & groupbit) {

          f[i][0] += mag_f[i][0];
          f[i][1] += mag_f[i][1];
          f[i][2] += mag_f[i][2];
        }
      }
    } 
    else if(moment_calc=="convergence" || moment_calc=="converge_check"){
      compute_magForce_converge();
    }
    else if(moment_calc=="linalg"){
      compute_magForce_linalg();
    }
    else {
      error->all(FLERR, "Invalid moment calculation method specidifed for fix magnetic");
    }
  } 
  else {
    error->all(FLERR, "Invalid model type specified for fix magnetic");
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

/* ----------------------------------------------------------------------
  Function to compute 3 X 3 matrix for moment_matrix 
------------------------------------------------------------------------- */

Eigen::Matrix3d FixMagnetic::Mom_Mat_ij(double sep_ij, Eigen::Vector3d SEP_ij_vec){
  double sep_pow3 = std::pow(sep_ij,3);
  double sep_pow5 = std::pow(sep_ij,5);

  double p4_sep_ij_pow5_div_3=(p4*sep_pow5)/3;
  double inv_p4_sep_ij_pow3=1/(p4*sep_pow3);

  // i-j 3 X 3 matrix definition
  Eigen::Matrix3d mom_mat_ij;
  mom_mat_ij = SEP_ij_vec*SEP_ij_vec.transpose()/p4_sep_ij_pow5_div_3 - Eigen::Matrix3d::Identity()*inv_p4_sep_ij_pow3;
  return mom_mat_ij;
}


/* ----------------------------------------------------------------------
  Function to compute separtion distance for a give particle pair
  4F vector with first three elements being the separation vector and last 
  one being its magnitude
  gives zero if they are not neighbors
------------------------------------------------------------------------- */

Eigen::VectorXd FixMagnetic::get_SEP_ij_vec(int x, int y) {
  
  int *jlist,*numneigh,**firstneigh;
  double *sepneigh, **firstsepneigh;
  
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;
  firstsepneigh = list->firstsepneigh;

  Eigen::VectorXd SEP_vec;
  SEP_vec = Eigen::VectorXd::Zero(4);

  int i = x;

  // Check if atom_id_1 is a local atom
  if (i < 0) {
    return SEP_vec; 
  }
  // Get the neighbor list for atom i
  jlist = firstneigh[i];
  int jnum = numneigh[i];
  sepneigh = firstsepneigh[i];

  // Iterate over the neighbors of atom i
  for (int jj = 0; jj < jnum; jj++) {
    int j = jlist[jj];
    j &= NEIGHMASK; // Apply neighbor mask (if necessary)

    // Check if the neighbor's atom ID matches atom_id_2
    if (j == y) {
      // std::cout<<"neighbor pair"<<std::endl<<std::endl;
      SEP_vec(0)=sepneigh[4*jj];
      SEP_vec(1)=sepneigh[4*jj+1];
      SEP_vec(2)=sepneigh[4*jj+2];
      SEP_vec(3)=std::sqrt(sepneigh[4*jj+3]);
      return SEP_vec; // Atoms are neighbors
    }
  }

  return SEP_vec; // Atoms are not neighbors
}


/* ----------------------------------------------------------------------
  Function to compute magnetic force using convergence method for MDM
------------------------------------------------------------------------- */

void FixMagnetic::compute_magForce_converge(){
  int i,ii,j,inum,totnum;
  int *ilist;
  double **mu = atom->mu;
  double **f = atom->f;
  double **mag_f = atom->mag_f;
  double *q = atom->q;
  int *mask = atom->mask;
  // int nlocal = atom->nlocal;
  // int nghost = atom->nghost;
  int *type = atom->type;

  // External magnetic field
  Eigen::Vector3d H0;
  H0<<ex,ey,ez;

  // // reallocate hfield array if necessary
  // if (varflag == ATOM && nlocal > maxatom) {
  //   maxatom = atom->nmax;
  //   memory->destroy(hfield);
  //   memory->create(hfield,maxatom,3,"hfield:hfield");
  // }

  // local atoms on this proc
  inum = list->inum;
  ilist = list->ilist;

  // radius and position structs for each atom
  rad = atom->radius;
  x = atom->x;


  if (varflag == CONSTANT) {

    /* ----------------------------------------------------------------------
      Moment Calculation
    ------------------------------------------------------------------------- */
    for (ii = 0; ii < inum; ii++) {
      // find the index for ii th local atom
      i = ilist[ii];

      if (mask[i] & groupbit) {

        // define vector for moment of ith particle
        Eigen::Vector3d mu_i_vector;

        // get susceptibility of particle ith particle
        double susc_i= susceptibility_[type[i]-1];
        double susc_eff_i=3*susc_i/(susc_i+3); // effective susceptibility

        // coefficient for ith particle
        double C_i = susc_eff_i*rad[i]*rad[i]*rad[i]*p4/3;

        // dipole moment based on FDM for ith particle
        mu_i_vector = C_i*H0;
        mu[i][0] = mu_i_vector[0];
        mu[i][1] = mu_i_vector[1];
        mu[i][2] = mu_i_vector[2];

        // loop over all other atoms in the system
        for (j=0; j<atom->natoms;j++){
          if(j==i) continue;

          // get susceptibility of particle jth particle
          double susc_j= susceptibility_[type[j]-1];
          double susc_eff_j=3*susc_j/(susc_j+3); // effective susceptibility

          // Calculate separation distance
          Eigen::Vector3d SEP_ij;
          double sep_ij;
          SEP_ij<<x[i][0]-x[j][0],x[i][1]-x[j][1],x[i][2]-x[j][2];
          sep_ij=SEP_ij.norm();

          // CHECK IF THE SEPARATION DISTANCE BETWEEN THE TWO PARTICLES
          // IS LOWER THAN THE SUM OF RADII.
          // CHANGE IT TO SUM OF RADII IF TRUE.
          
          // sum of radii of two particles
          double rad_sum_ij = rad[i] + rad[j];
          if (sep_ij/rad_sum_ij<1){
            SEP_ij=(SEP_ij/sep_ij)*rad_sum_ij;
            sep_ij=rad_sum_ij;
          }

          // moment of jth particle
          Eigen::Vector3d mu_j_vector;
          mu_j_vector << mu[j][0], mu[j][1], mu[j][2];  

          // H_dip due to particle jth at particle ith position
          Eigen::Vector3d H_dip_j;

          double mu_j_dot_sep= mu_j_vector.dot(SEP_ij/sep_ij);

          H_dip_j = (1/p4/sep_ij/sep_ij/sep_ij)*(3*mu_j_dot_sep*(SEP_ij/sep_ij) - mu_j_vector);

          // Modify dipole moment on ith particle due to jth particle
          mu_i_vector += C_i*H_dip_j;
        }

        // Check for convergence based on code by Tom
        if (moment_calc=="converge_check"){
          double mu_i_dot_mu_i;
          mu_i_dot_mu_i=mu_i_vector.dot(mu_i_vector);
          if (mu_i_dot_mu_i > (4*C_i*C_i*H0.dot(H0))){
            mu_i_vector = 2*C_i*H0.norm()*mu_i_vector/mu_i_vector.norm();
          }
        }
        mu[i][0]=mu_i_vector[0];
        mu[i][1]=mu_i_vector[1];
        mu[i][2]=mu_i_vector[2];
      }
    }
    /* ----------------------------------------------------------------------
      Force Calculation After Moment Calculation
    ------------------------------------------------------------------------- */
    for (ii = 0; ii < inum; ii++){
      // find the index for ii th local atom
      i = ilist[ii];

      if (mask[i] & groupbit) {
        
        // get susceptibility of ith particle
        double susc_i= susceptibility_[type[i]-1];
        // double susc_eff_i=3*susc_i/(susc_i+3); // effective susceptibility

        // get moment of ith particle
        Eigen::Vector3d mu_i_vector;
        mu_i_vector << mu[i][0], mu[i][1], mu[i][2];

        // Vectors for storing MDM, SHA and Total force for ith particle
        Eigen::Vector3d Force_mdm_i, Force_SHA_i, Force_tot_i;
        Force_mdm_i=Eigen::Vector3d::Zero();
        Force_SHA_i=Eigen::Vector3d::Zero();
        Force_tot_i=Eigen::Vector3d::Zero();

        // loop over all other atoms in the system
        for (j=0; j<atom->natoms;j++){
          if(j==i) continue; // skip for the same particle pair
        
          // get susceptibility of particle jth particle
          double susc_j= susceptibility_[type[j]-1];
          double susc_eff_j=3*susc_j/(susc_j+3); // effective susceptibility
          
          // get moment of jth particle
          Eigen::Vector3d mu_j_vector;
          mu_j_vector << mu[j][0], mu[j][1], mu[j][2];

          // calculate separation distance 
          Eigen::Vector3d SEP_ij;
          double sep_ij;
          SEP_ij<<x[i][0]-x[j][0],x[i][1]-x[j][1],x[i][2]-x[j][2];
          sep_ij=SEP_ij.norm();

          // CHECK IF THE SEPARATION DISTANCE BETWEEN THE TWO PARTICLES
          // IS LOWER THAN THE SUM OF RADII.
          // CHANGE IT TO SUM OF RADII IF TRUE.
          
          // sum of radii of two particles
          double rad_sum_ij = rad[i] + rad[j];
          if (sep_ij/rad_sum_ij<1){
            SEP_ij=(SEP_ij/sep_ij)*rad_sum_ij;
            sep_ij=rad_sum_ij;
          }

          // Calcualte MDM force between i-j pair
          Eigen::Vector3d Force_mdm_ij;
          double mu_i_dot_sep = mu_i_vector.dot(SEP_ij)/sep_ij;
          double mu_j_dot_sep = mu_j_vector.dot(SEP_ij)/sep_ij;
          double mu_i_dot_mu_j = mu_i_vector.dot(mu_j_vector);
          
          double sep_pow4 = std::pow(sep_ij,4);
          double K = 3*mu0/p4/sep_pow4;
          Force_mdm_ij = K*(mu_i_dot_sep*mu_j_vector+mu_j_dot_sep*mu_i_vector+(mu_i_dot_mu_j-5*mu_j_dot_sep*mu_i_dot_sep)*SEP_ij/sep_ij);

          // Add i-j MDM force to i the particle total MDM force
          Force_mdm_i += Force_mdm_ij;

          /* ----------------------------------------------------------------------
          SPHERICAL HARMONICS (if separation distance is less than x radii)
          ------------------------------------------------------------------------- */
          if (model_type == "inclusion"){
            if (sep_ij/rad[i] < 4){
              
              // Call sphertical harmonics class to calculate spherical harmonics
              spherical_harmonics SHA_i_j(rad[i], susc_i, H0, SEP_ij, mu_i_vector,mu_j_vector);

              Eigen::Vector3d Force_SHA_ij;
              Force_SHA_ij=SHA_i_j.get_force_actual_coord();

              Eigen::Vector3d Force_dip2B_ij;
              Force_dip2B_ij=SHA_i_j.get_force_2B_corrections();

              // add the spherical harmonics component and subtract the far-field affect (MDM)
              Force_SHA_i[0] += Force_SHA_ij[0] - Force_dip2B_ij[0];
              Force_SHA_i[1] += Force_SHA_ij[1] - Force_dip2B_ij[1];
              Force_SHA_i[2] += Force_SHA_ij[2] - Force_dip2B_ij[2];
            }
          }
        }
        // Total force
        Force_tot_i = Force_mdm_i + Force_SHA_i;

        // fix force for ith atom
        f[i][0] += Force_tot_i[0];
        f[i][1] += Force_tot_i[1];
        f[i][2] += Force_tot_i[2];

        // Store the computed forces
        mag_f[i][0] = Force_tot_i[0];
        mag_f[i][1] = Force_tot_i[1];
        mag_f[i][2] = Force_tot_i[2];
      }
    }
  }
}

/* ----------------------------------------------------------------------
  Function to compute magnetic force using linalg method for MDM
------------------------------------------------------------------------- */

void FixMagnetic::compute_magForce_linalg(){
  int i,j,k,ii,jj,inum,jnum;
  int *ilist,*jlist,*numneigh,**firstneigh,*slist;
  // double *sepneigh, **firstsepneigh;
  double **mu = atom->mu;
  double **f = atom->f;
  double **mag_f = atom->mag_f;
  double *q = atom->q;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  // int nghost = atom->nghost;
  int *type = atom->type;

  // External magnetic field
  Eigen::Vector3d H0;
  H0<<ex,ey,ez;

  // total number of atoms
  int natoms = atom->natoms;

  // lists for local atoms on this proc
  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;
  // firstsepneigh = list->firstsepneigh;
  
  // radius and position structs for each atom
  rad = atom->radius;
  x = atom->x;
  
  
  if (varflag == CONSTANT) {

    /* ----------------------------------------------------------------------
      Moment Calculation
    ------------------------------------------------------------------------- */

    /* ----------------------------------------------------------------------
      Compute Local Matrices
    ------------------------------------------------------------------------- */

    std::cout<< "STARTING MOMENT CALCULATION" <<std::endl;

    for (int i_proc = 0; i_proc < comm->nprocs; i_proc++) {
      if (comm->me == i_proc) {
        // std::cout<<"proc number"<<i_proc<<std::endl;
        // std::cout<<"recvcount [i_proc]"<<recvcounts[i_proc]<<"displs [i_proc]"<<displs[i_proc]<<std::endl;
        // for (int ii = 0; ii < natoms; ii++)
        // {
        //   std::cout<<"pos "<<ii<<": "<<x[ii][2]<<std::endl;
        // }
        std::cout<<"Processor"<<comm->me<<"pars"<<inum<<std::endl;
        // std::cout << "Process " << comm->me << ": my_variable = " << std::endl;
        // std::cout << local_matrix << std::endl;
      }
      MPI_Barrier(world); // Ensure synchronized printing
    }

    // Local matrices
    Eigen::MatrixXd local_matrix(3 * natoms, 3 * inum);
    local_matrix=Eigen::MatrixXd::Zero(3 * natoms, 3 * inum); 

    for (ii = 0; ii < inum; ii++){
      // find the index for ii th local atom
      i = ilist[ii];

      if (mask[i] & groupbit) {
        // get susceptibility of particle ith particle
        double susc_i= susceptibility_[type[i]-1];
        double susc_eff_i=3*susc_i/(susc_i+3); // effective susceptibility

        // coefficient for ith particle
        double C_i = susc_eff_i*rad[i]*rad[i]*rad[i]*p4/3;

        // find global index of ith particle
        int i_index = atom->tag[i];

        for (j = 0; j < natoms; j++){

          // find global index of ith particle
          int j_index = atom->tag[j];

          if ((i_index-1)!=(j_index-1)){
            // Calculate separation distance
            Eigen::Vector3d SEP_ij;
            double sep_ij;
            SEP_ij<<x[i][0]-x[j][0],x[i][1]-x[j][1],x[i][2]-x[j][2];
            sep_ij=SEP_ij.norm();
            
            // CHECK IF THE SEPARATION DISTANCE BETWEEN THE TWO PARTICLES
            // IS LOWER THAN THE SUM OF RADII.
            // CHANGE IT TO SUM OF RADII IF TRUE.
          
            // sum of radii of two particles
            double rad_sum_ij = rad[i] + rad[j];
            if (sep_ij/rad_sum_ij<1){
              SEP_ij=(SEP_ij/sep_ij)*rad_sum_ij;
              sep_ij=rad_sum_ij;
            }

            // i-j 3 X 3 matrix definition
            Eigen::Matrix3d mom_mat_ij;
            mom_mat_ij=Mom_Mat_ij(sep_ij, SEP_ij);

            local_matrix.block((j_index-1)*3,ii*3,3,3)=-mom_mat_ij;

          }

          // Calcuate matrix diagonals
          else if ((i_index-1)==(j_index-1)){
            local_matrix.block((j_index-1)*3,ii*3,3,3)=1/C_i*Eigen::Matrix3d::Identity();
          }
        }
      }
    }


    std::cout<< "Local moment matrix generated" <<std::endl;

    /* ----------------------------------------------------------------------
      Gather Local Matrices on proc 0 and find moments for each particle
    ------------------------------------------------------------------------- */

    // Receive counts and displacements for the data
    int *num_local = new int[comm->nprocs];
    int *recvcounts = new int[comm->nprocs];
    int *sendcounts = new int[comm->nprocs];
    int *displs_recv = new int[comm->nprocs];
    int *displs_send = new int[comm->nprocs];
    MPI_Allgather(&nlocal, 1, MPI_INT, num_local, 1, MPI_INT, world);


    displs_recv[0] = 0;
    for (int i_proc = 0; i_proc < comm->nprocs; i_proc++) {
      recvcounts[i_proc] = 3*3*natoms*num_local[i_proc];
      if (i_proc>0){
        displs_recv[i_proc] = displs_recv[i_proc - 1] + recvcounts[i_proc - 1];
      }
    }

    for (int i_proc = 0; i_proc < comm->nprocs; i_proc++) {
      if (comm->me == i_proc) {
        // std::cout<<"proc number"<<i_proc<<std::endl;
        // std::cout<<"recvcount [i_proc]"<<recvcounts[i_proc]<<"displs [i_proc]"<<displs[i_proc]<<std::endl;
        // for (int ii = 0; ii < natoms; ii++)
        // {
        //   std::cout<<"pos "<<ii<<": "<<x[ii][2]<<std::endl;
        // }
        std::cout<<"Processor"<<comm->me<<"pars"<<num_local[i_proc]<<std::endl;
        // std::cout << "Process " << comm->me << ": my_variable = " << std::endl;
        // std::cout << local_matrix << std::endl;
      }
      MPI_Barrier(world); // Ensure synchronized printing
    }
    

    // Defining the global moment_matrix
    Eigen::MatrixXd moment_matrix;
    if (comm->me==0){
      // 3N X 3N matrix for momemnt calculation where N is total number of particles
      moment_matrix= Eigen::MatrixXd(3*(natoms),3*(natoms));   
    }
    
    // Gather the data

    MPI_Gatherv(local_matrix.data(), local_matrix.size(), MPI_DOUBLE, moment_matrix.data(), recvcounts, displs_recv, MPI_DOUBLE, 0, world);

    // delete memory
    delete []recvcounts;
    delete []displs_recv;

    // define the moment_vec for all atoms
    Eigen::VectorXd mom_vec(3*(natoms));

    // use the matrix for linear solver on proc 0

    // moment_matrix * mom_vec = H_vec
    if (comm->me==0){
      
      std::cout<< "Global moment matrix gathered" <<std::endl;

      // H0 Vector
      Eigen::VectorXd H_vec(3*(natoms));
      H_vec = H0.replicate(natoms,1);
      
      // std::cout<<"Momemnt Matrix"<<std::endl;
      // std::cout<<moment_matrix<<std::endl<<std::endl<<std::endl;

      mom_vec=moment_matrix.colPivHouseholderQr().solve(H_vec);

      std::cout<< "Solved moments" <<std::endl;

      // std::cout<<"Moment Vector"<<mom_vec.transpose()<<std::endl<<std::endl<<std::endl;
    }

    /* ----------------------------------------------------------------------
      Distribute moments to processors
    ------------------------------------------------------------------------- */

    // Calculate sendcounts and displacements
    displs_send[0] = 0;

    for (int i_proc = 0; i_proc < comm->nprocs; i_proc++) {
      sendcounts[i_proc] = 3*num_local[i_proc];
      if (i_proc>0){
        displs_send[i_proc] = displs_send[i_proc - 1] + sendcounts[i_proc - 1];
      }      
    }

    // Create a receive buffer on each processor
    Eigen::VectorXd local_mom_vec(sendcounts[comm->me]); // Allocate on each processor

    // Scatter the vector
    MPI_Scatterv(mom_vec.data(), sendcounts, displs_send, MPI_DOUBLE, local_mom_vec.data(), sendcounts[comm->me], MPI_DOUBLE, 0, world);
    
    // delete memory
    delete []sendcounts;
    delete []displs_send;

    std::cout<< "local moments scattered" <<std::endl;
    
    // MPI_Bcast(mom_vec.data(), mom_vec.size(), MPI_DOUBLE, 0, world);

    // for (int i_proc = 0; i_proc < comm->nprocs; i_proc++) {
    //   if (comm->me == i_proc) {
    //     std::cout << "Process " << comm->me << ": my_variable = " << std::endl;
    //     std::cout << local_mom_vec.transpose() << std::endl;
    //     std::cout << "Displ " << displs_send[i_proc-1]<<std::endl;
    //     std::cout << "Sendcounts " << sendcounts[i_proc-1]<<std::endl;
    //   }
    //   MPI_Barrier(world); // Ensure synchronized printing
    // }
    
    for (ii = 0; ii < inum; ii++){
      // find the index for ii th local atom
      i = ilist[ii];
      // int i_index=atom->tag[i];
      if (mask[i] & groupbit) {

        Eigen::Vector3d mu_i_vector;

        mu_i_vector=local_mom_vec.segment(3*(ii),3);
        mu[i][0]=mu_i_vector[0];
        mu[i][1]=mu_i_vector[1];
        mu[i][2]=mu_i_vector[2];
      }      
    }
    
    std::cout<< "particle moment assigned" <<std::endl;
    
    /* ----------------------------------------------------------------------
      Force Calculation After Moment Calculation
    ------------------------------------------------------------------------- */
    for (ii = 0; ii < inum; ii++) {
      // find the index for ii th local particle
      i = ilist[ii];
    
      if (mask[i] & groupbit) {

        // get neighbor list for the ith particle
        jlist = firstneigh[i];
        jnum = numneigh[i];
        // sepneigh = firstsepneigh[i];

        // get susceptibility of particle ith particle
        double susc_i= susceptibility_[type[i]-1];
        // double susc_eff_i=3*susc_i/(susc_i+3); // effective susceptibility


        // get moment of ith particle
        Eigen::Vector3d mu_i_vector;
        mu_i_vector << mu[i][0], mu[i][1], mu[i][2];

        // Vectors for storing MDM, SHA and Total force for ith particle
        Eigen::Vector3d Force_mdm_i, Force_SHA_i, Force_tot_i;
        Force_mdm_i=Eigen::Vector3d::Zero();
        Force_SHA_i=Eigen::Vector3d::Zero();
        Force_tot_i=Eigen::Vector3d::Zero();

        // loop over each neighbor
        for (jj = 0; jj<jnum; jj++)  {
          j =jlist[jj];
          j &= NEIGHMASK;

          // get moment of jth particle
          Eigen::Vector3d mu_j_vector;
          mu_j_vector << mu[j][0], mu[j][1], mu[j][2];

          // get separation distance from the neighbor calculations
          Eigen::Vector3d SEP_ij;
          
          SEP_ij<<x[i][0]-x[j][0],x[i][1]-x[j][1],x[i][2]-x[j][2];
          double sep_ij=SEP_ij.norm();

          // CHECK IF THE SEPARATION DISTANCE BETWEEN THE TWO PARTICLES
          // IS LOWER THAN THE SUM OF RADII.
          // CHANGE IT TO SUM OF RADII IF TRUE.
          
          // sum of radii of two particles
          double rad_sum_ij = rad[i] + rad[j];
          if (sep_ij/rad_sum_ij<1){
            SEP_ij=(SEP_ij/sep_ij)*rad_sum_ij;
            sep_ij=rad_sum_ij;
          }

          // Calcualte MDM force between i-j pair
          Eigen::Vector3d Force_mdm_ij;
          double mu_i_dot_sep = mu_i_vector.dot(SEP_ij)/sep_ij;
          double mu_j_dot_sep = mu_j_vector.dot(SEP_ij)/sep_ij;
          double mu_i_dot_mu_j = mu_i_vector.dot(mu_j_vector);
          
          double sep_pow4 = std::pow(sep_ij,4);
          double K = 3*mu0/p4/sep_pow4;
          
          Force_mdm_ij = K*(mu_i_dot_sep*mu_j_vector+mu_j_dot_sep*mu_i_vector+(mu_i_dot_mu_j-5*mu_j_dot_sep*mu_i_dot_sep)*SEP_ij/sep_ij);
          
          // Add i-j MDM force to i the particle total MDM force
          Force_mdm_i += Force_mdm_ij;

          /* ----------------------------------------------------------------------
          SPHERICAL HARMONICS (if separation distance is less than x radii)
          ------------------------------------------------------------------------- */
          if (model_type == "inclusion"){
            if (sep_ij/rad[i] < 4){

              spherical_harmonics SHA_i_j(rad[i], susc_i, H0, SEP_ij, mu_i_vector,mu_j_vector);

              Eigen::Vector3d Force_SHA_ij;
              Force_SHA_ij=SHA_i_j.get_force_actual_coord();

              Eigen::Vector3d Force_dip2B_ij;
              Force_dip2B_ij=SHA_i_j.get_force_2B_corrections();

              // add the spherical harmonics component and subtract the far-field affect (MDM)
              Force_SHA_i[0] += Force_SHA_ij[0] - Force_dip2B_ij[0];
              Force_SHA_i[1] += Force_SHA_ij[1] - Force_dip2B_ij[1];
              Force_SHA_i[2] += Force_SHA_ij[2] - Force_dip2B_ij[2];
            }
          }
        }
        // Total force
        Force_tot_i = Force_mdm_i + Force_SHA_i;

        // fix force for ith atom
        f[i][0] += Force_tot_i[0];
        f[i][1] += Force_tot_i[1];
        f[i][2] += Force_tot_i[2];

        // Store the computed forces
        mag_f[i][0] = Force_tot_i[0];
        mag_f[i][1] = Force_tot_i[1];
        mag_f[i][2] = Force_tot_i[2];
      }
    }
  }
}