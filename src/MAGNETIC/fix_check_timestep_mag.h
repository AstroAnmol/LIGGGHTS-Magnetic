
/*
 *  fix_check_timestep.h
 *
 *      Author: ASikka
 */

#ifdef FIX_CLASS

FixStyle(check/timestep/mag,FixCheckTimestepMag)

#else

#ifndef LMP_FIX_CHECK_TIMESTEP_MAG_H
#define LMP_FIX_CHECK_TIMESTEP_MAG_H

#include "fix.h"

namespace LAMMPS_NS {

class FixCheckTimestepMag : public Fix {
 public:
  FixCheckTimestepMag(class LAMMPS *, int, char **);
  int setmask();
  void init();
  void end_of_step();
//   double compute_vector(int);

 private:
  const double mu0 = 4.0 * M_PI * 1e-7;
  double dF_by_F;
  class Properties* properties;
  class PairGran* pg;
  class FixWallGran* fwg;
  class FixPropertyGlobal* susceptibility;
  void calc_magnetic_estims();
  bigint mag_timestep_value; // wait these many timesteps before computing force again
  double magnetic_time;
  double fraction_magnetic;
  double fraction_magnetic_lim;
  double vmax_user;
  double v_rel_max_sim; //max relative velocity detected in simulation

  bool warnflag,errorflag;
  double MagForceEmperical(double susc);
  double d_MagForceEmperical(double susc);
};

}

#endif
#endif
