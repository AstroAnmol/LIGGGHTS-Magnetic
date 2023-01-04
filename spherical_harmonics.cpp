#include <iostream>
#include <cmath>
#include "spherical_harmonics.h" 
#include <eigen-3.4.0/Eigen/Dense>
#define __STDCPP_WANT_MATH_SPEC_FUNCS__
// Intiator
spherical_harmonics::spherical_harmonics(double radius, double susceptibilty, Eigen::Vector3d H0_vec, Eigen::Vector3d SEP_vec, Eigen::Vector3d M_i_vec){
    // variable assignment
    a=radius;
    susc=susceptibilty;
    H0=H0_vec*mu0;
    SEP=SEP_vec;
    M_i=M_i_vec;

    L=10; //choose option later
    double mu=(1+susc)*mu0;
    susc_eff=3*susc/(susc+3);
    sep = SEP.norm();

    // axis definition
    z_cap=SEP/sep;

    x_cap=(H0-H0.dot(z_cap)*z_cap);

    if (x_cap.norm()==0){
      // how to define the axis in this case
    }
    else{
        x_cap=x_cap/x_cap.norm();

        y_cap=z_cap.cross(x_cap);

    }

    double H_prll, H_perp;

    H_prll=H0.dot(z_cap);
    H_perp=H0.dot(x_cap);

    std::cout<<H_prll<< H_perp<<std::endl;

    for (int m= 0; m < 2; m++){
        Eigen::MatrixXd X(L,L), Delta_m(L,L), Gamma_m(L,L); 
        for (int i = 0; i < L; i++){
            for (int j = 0; j < L; j++){
                // X matrix
                if (i==j)
                {
                    X(i,j)=(i+1)*(mu/mu0) + (i+1) + 1;
                }
                // Delta and Gamma matrix
                Delta_m(i,j)=std::pow((-1),((i+1)+m))*((i+1)*(mu/mu0)-(i+1))*nchoosek(i+1+j+1, j+1-m)*std::pow(a,(2*(i+1)+1))/std::pow(sep,(i+1+j+1+1));
                Gamma_m(i,j)=std::pow((-1), (i+1+j+1))*Delta_m(i,j);
            }   
        }
        // 2L X 2L Matrix
        Eigen::MatrixXd Am(2*L, 2*L);
        Am.block(0,0,L,L)=X;
        Am.block(0,L,L,L)=Delta_m;
        Am.block(L,0,L,L)=Gamma_m;
        Am.block(L,L,L,L)=X;

        //qm vector
        Eigen::VectorXd qm(L);
        if (m==0){
            qm(0)=-H_prll*std::pow(a,3)*(1-mu/mu0);
        }
        else if (m==1){
            qm(0)=H_perp*std::pow(a,3)*(1-mu/mu0);
        }
        
        //2L Q vector
        Eigen::VectorXd Qm(2*L);
        Qm.block(0,0,L,1)=qm;
        Qm.block(L,0,L,1)=qm;

        //solve linear system
        Eigen::VectorXd Beta_m(2*L);

        Beta_m=Am.colPivHouseholderQr().solve(Qm);
        if (m==0){
            Beta1_0=Beta_m.block(0,0,L,1);
            Beta2_0=Beta_m.block(L,0,L,1);
        }
        else if (m==1){
            Beta1_1=Beta_m.block(0,0,L,1);
            Beta2_1=Beta_m.block(L,0,L,1);
        }
    };

    std::cout<<Beta1_0<< Beta1_1<<std::endl;
    // std::cout<< "Linear System Solved"<<std::endl;
    
    //adjust two-body dipole moments
    double Beta_01_dip=  M_i.dot(z_cap)/(4*M_PI*a*a*a);
    double Beta_11_dip= -M_i.dot(x_cap)/(4*M_PI*a*a*a);

    //
    M_dipole = 4*M_PI*a*a*a*susc_eff*H0/3;
    double Beta_01_2Bdip = M_dipole.dot(z_cap)/(4*M_PI*a*a*a);
    double Beta_11_2Bdip = -M_dipole.dot(x_cap)/(4*M_PI*a*a*a);

    Beta1_0[0]=Beta1_0[0] + Beta_01_dip - Beta_01_2Bdip;
    Beta2_0[0]=Beta2_0[0] + Beta_01_dip - Beta_01_2Bdip;
    Beta1_1[0]=Beta1_1[0] + Beta_11_dip - Beta_11_2Bdip;
    Beta2_1[0]=Beta2_1[0] + Beta_11_dip - Beta_11_2Bdip;

    // Create a 3D spherical mesh
    int N =180;
    double dang= M_PI/N;
    Eigen::VectorXd inc= Eigen::VectorXd::LinSpaced(N+1,dang/2, M_PI + dang/2).transpose();
    Eigen::VectorXd az= Eigen::VectorXd::LinSpaced(2*N+1,dang/2, 2*M_PI + dang/2).transpose();


    F=Eigen::Vector3d::Zero();

    // Formulating the Maxwell Stress Tensor in Spherical Coordinates
    for (int i = 0; i < 2*N+1; i++){
        double p;
        if (i==0 or i==(2*N)){
            p=1;}
        else if (i%2==0){
            p=2;}
        else{ p=4;}
        for (int j = 0; j < N+1; j++){
            double q;
            if (j==0 or j==(N)){
                q=1;}
            else if (j%2==0){
                q=2;}
            else{ q=4;}
            double ph= az[i];
            double th= inc[j];
            F=F+ a*p*q*integrand(th, ph);
        }
    }
    F=F*dang*dang/9.0;

    F_act_coord=F[0]*x_cap + F[1]*y_cap + F[2]*z_cap;
}

Eigen::Vector3d spherical_harmonics::get_force(){
    return F;
}

Eigen::Vector3d spherical_harmonics::get_force_actual_coord(){
    return F_act_coord;
}

double spherical_harmonics::nchoosek(int n, int k){
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

Eigen::Vector3d spherical_harmonics::mag_field(double r, double theta, double phi){
    double Hr=0, Hth=0, Hphi=0;
    for (int l = 1; l < L+1; l++){
        for (int m = 0; m < 2; m++){
            double Hrs=0, Hths=0, Hphis=0;
            for (int s = m; s < L+1; s++){
                double Psm=lpmn_cos(m, s, theta);
                double dPsm=d_lpmn_cos(m, s, theta);
                double ls_choose_sm=nchoosek(l+s, s+m);
                double r_pow_s1=std::pow(r, (s-1));
                double sep_pow=std::pow(sep, l+s+1);
                double minus_one_pow=std::pow(-1, s+m);
                double r_pow_times_Psm=r_pow_s1*Psm;
                
                // r component
                double additional=(minus_one_pow)*s*r_pow_times_Psm/(sep_pow);
                Hrs=Hrs + additional*(ls_choose_sm);
                // theta component
                Hths=Hths + (minus_one_pow)*ls_choose_sm*(r_pow_s1*dPsm)/(sep_pow);
                // phi component
                if (m==1){
                    Hphis= Hphis + (minus_one_pow)*ls_choose_sm*r_pow_times_Psm/(std::sin(theta))/(sep_pow);
                }
            }
            double Plm=lpmn_cos(m, l, theta);
            double dPlm=d_lpmn_cos(m, l, theta);
            double r_pow_l2=std::pow(r, l+2);
            // std::cout<<"Hrs"<<Hrs<<"Hths"<<Hths<<std::endl;
            if (m==0){
                // R component
                Hr=Hr + (((l+1)*Beta1_0[l-1]*(Plm/r_pow_l2) -  Beta2_0[l-1]*Hrs)*std::cos(m*phi));
                // Theta component
                Hth=Hth + ((Beta1_0[l-1]*(dPlm/r_pow_l2) + Beta2_0[l-1]*Hths)*std::cos(m*phi));
            }
            else if (m==1){
                // R Component
                Hr=Hr + (((l+1)*Beta1_1[l-1]*(Plm/r_pow_l2) - Beta2_1[l-1]*Hrs)*std::cos(m*phi));
                // Theta component
                Hth=Hth + ((Beta1_1[l-1]*(dPlm/r_pow_l2) + Beta2_1[l-1]*Hths)*std::cos(m*phi));
                // Phi component
                Hphi=Hphi + ((Beta1_1[l-1]*(Plm/(std::sin(theta)/r_pow_l2)) + Beta2_1[l-1]*Hphis)*std::sin(phi));
            }
        }
    }
    Hth=-Hth;
    Eigen::Vector3d magfield;
    magfield << Hr, Hth, Hphi;
    return magfield;
}

//integrand function
Eigen::Vector3d spherical_harmonics::integrand(double th, double ph){
    //transformation matrix
    Eigen::Matrix3d pre, post, T_cart;
    pre<<   std::sin(th)*std::cos(ph), std::cos(th)*std::cos(ph), -std::sin(ph),
            std::sin(th)*std::sin(ph), std::cos(th)*std::sin(ph),  std::cos(ph),
            std::cos(th), -std::sin(th),  0;
    post= pre.transpose();
    Eigen::Vector3d H0_sph, H_sph, H_cart, rn_hat;
    H0_sph=post*H0;
    H_sph= mag_field(a, th, ph) + H0_sph;
    H_cart=pre*H_sph;
    H_cart[1]=H_cart[1]-(M_i.dot(y_cap)/(4*M_PI*a*a*a))*(lpmn_cos(1,1, th)*std::sin(ph)/(a*a));
    double h=H_cart.norm();
    T_cart=mu0*(H_cart*H_cart.transpose() - 0.5*(h*h)*Eigen::Matrix3d::Identity());
    rn_hat<<std::sin(th)*std::cos(ph),std::sin(th)*std::sin(ph),std::cos(th);
    return std::sin(th)*T_cart*rn_hat;
}

//define associate legendre functions for cos(theta)
double spherical_harmonics::lpmn_cos(int m, int n, double theta){
    return std::pow(-1,m)*std::assoc_legendre(n, m, std::cos(theta));
}

double spherical_harmonics::d_lpmn_cos(int m, int n, double theta){
    return ((m-n-1)*lpmn_cos(m,n+1,theta) + (n+1)*std::cos(theta)*lpmn_cos(m,n,theta))/(-std::sin(theta));
}