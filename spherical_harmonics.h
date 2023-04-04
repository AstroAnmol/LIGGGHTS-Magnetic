#ifndef SPHERICAL_HARMONICS_H
#define SPHERICAL_HARMONICS_H
#include <cmath>
#include <eigen-3.4.0/Eigen/Dense>

class spherical_harmonics{
    private:
        //constants
        double a, sep; //radius of particles (m) and separation between particles (m)
        double susc; // magnetic susceptibility
        double susc_eff; //effective susceptibility
        double hmag; // magneitc field mag (A/m)
        double mu0=4*M_PI*1e-07;
        int L; // number of multipoles used
        Eigen::Vector3d H0;
        Eigen::Vector3d SEP;
        Eigen::Vector3d z_cap, x_cap, y_cap;
        Eigen::VectorXd Beta1_0, Beta2_0, Beta1_1, Beta2_1;
        Eigen::Vector3d M_i, M_dipole, M_j;
        Eigen::Vector3d F;
        Eigen::Vector3d F_act_coord;
        //functions
        double nchoosek(int n, int k);
        double lpmn_cos(int n, int m, double theta);
        double d_lpmn_cos(int n, int m, double theta);
    public:
        spherical_harmonics(double radius, double susceptibilty, Eigen::Vector3d H0_vec, Eigen::Vector3d SEP_vec, Eigen::Vector3d M_i_vec, Eigen::Vector3d M_j_vec); // array for magnetic force parameters [a, susc]
        Eigen::Vector3d integrand(double th, double ph);
        Eigen::Vector3d mag_field(double r, double theta, double phi);
        Eigen::Vector3d get_force();
        Eigen::Vector3d get_force_actual_coord();
};
#endif