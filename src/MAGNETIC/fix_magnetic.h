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
  int varflag;
  char *xstr,*ystr,*zstr;
  int xvar,yvar,zvar,xstyle,ystyle,zstyle;
  int nlevels_respa;
  class NeighList *list;
  // double nchoosek(int n, int k);
  int maxatom;
  double **hfield;

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
