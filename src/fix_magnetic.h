/*
 * fix_magnetic.h
 *
 *      Author: ASikka
 */

#ifdef FIX_CLASS

FixStyle(magnetic,FixMagnetic)

#else

#ifndef LMP_FIX_MAGNETIC_H
#define LMP_FIX_MAGNETIC_H

#include "fix.h"
#include <Eigen/Dense>

namespace LAMMPS_NS {

class FixMagnetic : public Fix {
 public:
  FixMagnetic(class LAMMPS *, int, char **);
  ~FixMagnetic();
  int setmask();
  void init();
  void init_list(int, class NeighList *);
  void setup(int);
  void post_force(int);
  void post_force_respa(int, int, int);
  double memory_usage();

 private:
  double ex,ey,ez;
  double *rad;
  double **x;
  int *atom_id;
  int varflag;
  char *xstr,*ystr,*zstr;
  int xvar,yvar,zvar,xstyle,ystyle,zstyle;
  int nlevels_respa;
  class NeighList *list;
  // double nchoosek(int n, int k);
  int maxatom;
  double **hfield;
  double **last_forces; // Store last computed forces
  bigint N_magforce_timestep; // wait these many timesteps before computing force again
  // bigint last_magforce_timestep; // timestep when magnetic forces were last computed

  /* ----------------------------------------------------------------
  variables and functions needed for mag force calculation
  ----------------------------------------------------------------- */ 

  // variables
  double p4 = M_PI*4;
  double mu0 = p4*1e-7;
  Eigen::MatrixXd SEP_x_mat, SEP_y_mat, SEP_z_mat, sep_mat;
  Eigen::MatrixXd sep_pow3, sep_pow4, sep_pow5;

  //functions
  Eigen::Matrix3d Mom_Mat_ij(int i, int j);
  void compute_SEP(int i, int j);



protected:
  class FixPropertyGlobal* fix_susceptibility_;
  double *susceptibility_;

};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Fix efield requires atom attribute q

Self-explanatory.

E: Variable name for fix efield does not exist

Self-explanatory.

E: Variable for fix efield is invalid style

Only equal-style variables can be used.

*/
