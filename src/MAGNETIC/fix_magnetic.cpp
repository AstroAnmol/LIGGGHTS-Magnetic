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

// #include <iostream>
#include "mpi_liggghts.h"
#include "comm.h"

#define EIGEN_DONT_PARALLELIZE

using namespace LAMMPS_NS;
using namespace FixConst;

enum{CONSTANT,EQUAL,ATOM};

/* ---------------------------------------------------------------------- */

FixMagnetic::FixMagnetic(LAMMPS *lmp, int narg, char **arg) :

  Fix(lmp, narg, arg),
  N_magforce_timestep(0),
  fix_susceptibility_(0),
  susceptibility_(0)
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
  warnflag = true;
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

  // pre-calculate susceptibility 
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
    else if(moment_calc=="convergence"){
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

  double p4_sep_ij_pow5_div_3=(P4*sep_pow5)/3;
  double inv_p4_sep_ij_pow3=1/(P4*sep_pow3);

  // i-j 3 X 3 matrix definition
  Eigen::Matrix3d mom_mat_ij;
  mom_mat_ij = SEP_ij_vec*SEP_ij_vec.transpose()/p4_sep_ij_pow5_div_3 - Eigen::Matrix3d::Identity()*inv_p4_sep_ij_pow3;
  return mom_mat_ij;
}

/* ----------------------------------------------------------------------
  Function to compute magnetic force using convergence method for MDM
------------------------------------------------------------------------- */

void FixMagnetic::compute_magForce_converge(){
  int i,j,ii,jj,inum,jnum;
  int *ilist,*jlist,*numneigh,**firstneigh;
  double **mu = atom->mu;
  double **f = atom->f;
  double **mag_f = atom->mag_f;
  // double *q = atom->q;
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
  
  // radius and position structs for each atom
  rad = atom->radius;
  x = atom->x;

  if (varflag == CONSTANT) {

    /* ----------------------------------------------------------------------
      Moment Calculation
       First for loop to converge over number of steps
    ------------------------------------------------------------------------- */

    // Maximum number of steps 
    int max_step = 20;
    // ith step
    int i_step;
    // epsilon for convergence
    double epsilon = 1e-6;

     // Local matrices
    Eigen::MatrixXd local_matrix;
    Eigen::VectorXd local_Ci;
    Eigen::VectorXi local_index;

    local_index=Eigen::VectorXi::Zero(inum);
    local_Ci=Eigen::VectorXd::Zero(inum);
    local_matrix=Eigen::MatrixXd::Zero(3 * natoms, 3 * inum);

    for (ii = 0; ii < inum; ii++){
      // find the index for ii th local atom
      i = ilist[ii];

      if (mask[i] & groupbit) {

        // define vector for moment of ith particle
        Eigen::Vector3d mu_i_vector;

        // get susceptibility of particle ith particle
        double susc_i= susceptibility_[type[i]-1];
        double susc_eff_i=3*susc_i/(susc_i+3); // effective susceptibility

        // coefficient for ith particle
        double C_i = susc_eff_i*rad[i]*rad[i]*rad[i]*P4/3;

        // find global index of ith particle
        int i_index = atom->tag[i];
        
        // get neighbor list for the ith particle
        jlist = firstneigh[i];
        jnum = numneigh[i];

        // go over all the neighbors for non-diagnonal matrices
        for (jj = 0; jj < jnum; jj++){

          j=jlist[jj];

          // find global index of ith particle
          int j_index = atom->tag[j];
          
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

          local_matrix.block((j_index-1)*3,ii*3,3,3)=C_i*mom_mat_ij;
          local_Ci(ii)=C_i;
          local_index(ii)=i_index;
        }
      }
    }
    

    /* ----------------------------------------------------------------------
      Gather Local Matrices on proc 0 and find moments for each particle
    ------------------------------------------------------------------------- */

    // Receive counts and displacements for the data
    int *num_local = new int[comm->nprocs];
    int *recvcounts = new int[comm->nprocs];
    int *sendcounts = new int[comm->nprocs];
    int *displs_recv = new int[comm->nprocs];
    int *displs_send = new int[comm->nprocs];
    int *recvcounts_id = new int[comm->nprocs];
    int *displs_id = new int[comm->nprocs];
    MPI_Allgather(&nlocal, 1, MPI_INT, num_local, 1, MPI_INT, world);


    displs_recv[0] = 0;
    displs_id[0] = 0;
    for (int i_proc = 0; i_proc < comm->nprocs; i_proc++) {
      recvcounts[i_proc] = 3*3*natoms*num_local[i_proc];
      recvcounts_id[i_proc] = num_local[i_proc]; 
      if (i_proc>0){
        displs_recv[i_proc] = displs_recv[i_proc - 1] + recvcounts[i_proc - 1];
        displs_id[i_proc] = displs_id[i_proc - 1] + recvcounts_id[i_proc - 1];
      }
    }

    // Defining the global moment_matrix
    Eigen::MatrixXd moment_matrix;
    Eigen::VectorXi column_index;
    Eigen::VectorXd global_Ci;
    if (comm->me==0){
      // 3N X 3N matrix for momemnt calculation where N is total number of particles
      moment_matrix= Eigen::MatrixXd(3*(natoms),3*(natoms)); 
      column_index= Eigen::VectorXi(natoms);
      global_Ci= Eigen::VectorXd(natoms);
    }
    
    // Gather the data

    MPI_Gatherv(local_matrix.data(), local_matrix.size(), MPI_DOUBLE, moment_matrix.data(), recvcounts, displs_recv, MPI_DOUBLE, 0, world);

    MPI_Gatherv(local_index.data(), local_index.size(), MPI_INT, column_index.data(), recvcounts_id, displs_id, MPI_INT, 0, world);

    MPI_Gatherv(local_Ci.data(), local_Ci.size(), MPI_DOUBLE, global_Ci.data(), recvcounts_id, displs_id, MPI_DOUBLE, 0, world);

    // delete memory
    delete []recvcounts;
    delete []displs_recv;
    delete []recvcounts_id;
    delete []displs_id;

    // define the moment_vec for all atoms
    Eigen::VectorXd mom_vec(3*(natoms));

    // run the convergence algorithm on proc 0
    if (comm->me==0){
      
      Eigen::MatrixXd A(3*natoms, 3*natoms);
      A = Eigen::MatrixXd::Zero(3*natoms,3*natoms);


      // arrange the global matrix acc to global index
      for (int k = 0; k < natoms; k++){
        A.block(0,(column_index(k)-1)*3,3*natoms,3) = moment_matrix.block(0,3*k,3*natoms,3);
      }

      Eigen::VectorXd mu_array(3*natoms);
      Eigen::VectorXd mu_diff(natoms);

      // Covergence loop
      for (i_step = 0; i_step<max_step; i_step++){

        for (int i = 0; i < natoms; i++){
          
          double Ci= global_Ci(column_index(i)-1);

          Eigen::Vector3d mu_i_vec;
          mu_i_vec = Ci*H0;

          for (int j = 0; j < natoms; j++){

            if (i!=j){

              Eigen::Vector3d mu_j_vec;
              mu_j_vec = mu_array.segment(3*j,3);

              // Calculate H_dip
              Eigen::Vector3d C_i_H_dip = A.block(3*i,3*j,3,3)*mu_j_vec;
              mu_i_vec+=C_i_H_dip;
            }
          }
          Eigen::Vector3d diff=mu_i_vec-mu_array.segment(3*i,3);
          mu_diff(i)=diff.norm()/H0.norm();
          mu_array.segment(3*i,3)=mu_i_vec;
        }
        
        // break the loop if convergence condition reached
        if (mu_diff.maxCoeff()<epsilon){
          break;
        }
        
      }
      
      // change the moment vector to local proc indices
      for (int k = 0; k < natoms; k++){
        mom_vec.segment(3*k,3) = mu_array.segment(3*(column_index(k)-1),3);
      }

      char errstr[512];

      if(i_step >= max_step){   
        sprintf(errstr,"Moment Calculation did not Converge after %d steps",max_step);
        error->warning(FLERR,errstr);
       }
      else{
        sprintf(errstr,"Moment Calculation converged after %d steps",i_step);
        error->warning(FLERR,errstr);
      }
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
    delete []num_local;
   
    // Set moments to atom variables
    for (ii = 0; ii < inum; ii++){
      // find the index for ii th local atom
      i = ilist[ii];
      if (mask[i] & groupbit) {

        // // get susceptibility of particle ith particle
        // double susc_i= susceptibility_[type[i]-1];
        // double susc_eff_i=3*susc_i/(susc_i+3); // effective susceptibility

        // // coefficient for ith particle
        // double C_i = susc_eff_i*rad[i]*rad[i]*rad[i]*P4/3;

        Eigen::Vector3d mu_i_vector;

        mu_i_vector=local_mom_vec.segment(3*(ii),3);
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

        // get neighbor list for the ith particle
        jlist = firstneigh[i];
        jnum = numneigh[i];

        // loop over all the neighbors in the system
        for (jj=0; jj<jnum;jj++){
          j=jlist[jj];          

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
          double K = 3*MU0/P4/sep_pow4;
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
  int i,j,ii,jj,inum,jnum;
  int *ilist,*jlist,*numneigh,**firstneigh;
  double **mu = atom->mu;
  double **f = atom->f;
  double **mag_f = atom->mag_f;
  // double *q = atom->q;
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

    // Local matrices
    Eigen::MatrixXd local_matrix;

    local_matrix=Eigen::MatrixXd::Zero(3 * natoms, 3 * inum);

    for (ii = 0; ii < inum; ii++){
      // find the index for ii th local atom
      i = ilist[ii];

      if (mask[i] & groupbit) {
        // get susceptibility of particle ith particle
        double susc_i= susceptibility_[type[i]-1];
        double susc_eff_i=3*susc_i/(susc_i+3); // effective susceptibility

        // coefficient for ith particle
        double C_i = susc_eff_i*rad[i]*rad[i]*rad[i]*P4/3;

        // find global index of ith particle
        int i_index = atom->tag[i];

        // Calcuate matrix diagonals
        local_matrix.block((i_index-1)*3,ii*3,3,3)=1/C_i*Eigen::Matrix3d::Identity();
        
        // get neighbor list for the ith particle
        jlist = firstneigh[i];
        jnum = numneigh[i];

        // go over all the neighbors for non-diagnonal matrices
        for (jj = 0; jj < jnum; jj++){

          j=jlist[jj];

          // find global index of ith particle
          int j_index = atom->tag[j];
          
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
      }
    }


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

      // H0 Vector
      Eigen::VectorXd H_vec(3*(natoms));
      H_vec = H0.replicate(natoms,1);

      mom_vec=moment_matrix.colPivHouseholderQr().solve(H_vec);
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
    delete []num_local;
   
    
    for (ii = 0; ii < inum; ii++){
      // find the index for ii th local atom
      i = ilist[ii];
      if (mask[i] & groupbit) {

        Eigen::Vector3d mu_i_vector;

        mu_i_vector=local_mom_vec.segment(3*(ii),3);
        mu[i][0]=mu_i_vector[0];
        mu[i][1]=mu_i_vector[1];
        mu[i][2]=mu_i_vector[2];
      }      
    }
    
    
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
          double K = 3*MU0/P4/sep_pow4;
          
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