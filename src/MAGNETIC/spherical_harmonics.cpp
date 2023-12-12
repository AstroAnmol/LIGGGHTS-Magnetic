#include <iostream>
#include <cmath>
// #include <fstream>
#include "spherical_harmonics.h" 
#include <Eigen/Dense>
#include <boost/math/special_functions/legendre.hpp>
// #define EIGEN_DONT_PARALLELIZE

#define __STDCPP_WANT_MATH_SPEC_FUNCS__


// Intiator
spherical_harmonics::spherical_harmonics(double radius, double susceptibilty, Eigen::Vector3d H0_vec, Eigen::Vector3d SEP_vec, Eigen::Vector3d M_i_vec, Eigen::Vector3d M_j_vec){


    // Eigen::initParallel();
    // variable assignment
    a=radius;
    susc=susceptibilty;
    H0=H0_vec;
    SEP=SEP_vec;
    M_i=M_i_vec;
    M_j=M_j_vec;

    if (susc<4){
        L=10;}
    else if (susc>=4 && susc<7){
        L=20;}
    else if (susc>=7 && susc<12){
        L=30;}
    else if (susc>=12 && susc<20){
        L=40;}
    else if (susc>=20 && susc<35){
        L=45;}
    else if (susc>=35 && susc<50){
        L=50;
    }

    double mu=(1+susc)*mu0;
    susc_eff=3*susc/(susc+3);
    sep = SEP.norm();
    double sep_sq = sep*sep;

    // std::cout<< "Initialization Done" <<std::endl<<std::endl;

    // axis definition
    z_cap=SEP/sep;

    x_cap=(H0-H0.dot(z_cap)*z_cap);

    if (x_cap.norm()==0){
        x_cap<<z_cap[2], 0, -z_cap[0];
        y_cap=z_cap.cross(x_cap);
    }
    else{
        x_cap=x_cap/x_cap.norm();

        y_cap=z_cap.cross(x_cap);
    }


    H_prll=H0.dot(z_cap);
    H_perp=H0.dot(x_cap);

    std::cout<< "H perp :" << H_perp <<std::endl<<std::endl;
    std::cout<< "H prll :" << H_prll <<std::endl<<std::endl;

    for (int m= 0; m < 2; m++){
        Eigen::MatrixXd X(L,L), Delta_m(L,L), Gamma_m(L,L); 
        for (int i = 0; i < L; i++){
            for (int j = 0; j < L; j++){
                // std::cout<< "Entered the innermost for loop" <<std::endl<<std::endl;
                // X matrix
                if (i==j){   
                    X(i,j)=(i+1)*(mu/mu0) + (i+1) + 1;
                    // std::cout<<X(i,j)<<std::endl;
                }
                else{   X(i,j)=0;}
                
                // Delta and Gamma matrix
                Delta_m(i,j)=std::pow((-1),((i+1)+m))*((i+1)*(mu/mu0)-(i+1))*nchoosek(i+1+j+1, j+1-m)*std::pow(a,(2*(i+1)+1))/std::pow(sep,(i+1+j+1+1));
                Gamma_m(i,j)=std::pow((-1), (i+1+j+1))*Delta_m(i,j);
            }
            // std::cout<< "Ended the innermost for loop" <<std::endl<<std::endl;   
        }

        // std::cout<< "Started making A matrix" <<std::endl<<std::endl;
        // 2L X 2L Matrix
        Eigen::MatrixXd Am(2*L, 2*L);
        Am.block(0,0,L,L)=X;
        Am.block(0,L,L,L)=Delta_m;
        Am.block(L,0,L,L)=Gamma_m;
        Am.block(L,L,L,L)=X;

        //qm vector
        Eigen::VectorXd qm(L);
        qm=Eigen::VectorXd::Zero(L);
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


    // std::cout<< "Solved linear equation loop" <<std::endl<<std::endl;

    // std::cout<<Beta1_0<<std::endl<<std::endl;

    //adjust two-body dipole moments
    double Beta_01_dip=  M_i.dot(z_cap)/(4*M_PI*a*a*a);
    double Beta_11_dip= -M_i.dot(x_cap)/(4*M_PI*a*a*a);

    double Beta_02_dip=  M_j.dot(z_cap)/(4*M_PI*a*a*a);
    double Beta_12_dip= -M_j.dot(x_cap)/(4*M_PI*a*a*a);

    Eigen::Matrix2d A0(2, 2), A1(2, 2);

    A0<<    mu/mu0 + 2 , (-1)*(mu/mu0-1)*nchoosek(2,1)*std::pow(a,3)/std::pow(sep,3),
            (-1)*(mu/mu0-1)*nchoosek(2,1)*std::pow(a,3)/std::pow(sep,3),  mu/mu0 + 2;

    A1<<    mu/mu0 + 2 , (mu/mu0-1)*nchoosek(2,2)*std::pow(a,3)/std::pow(sep,3),
            (mu/mu0-1)*nchoosek(2,2)*std::pow(a,3)/std::pow(sep,3),  mu/mu0 + 2;
    
    double Beta_01_2Bdip, Beta_02_2Bdip, Beta_11_2Bdip, Beta_12_2Bdip;
    
    Eigen::Vector2d Q0, Q1, Beta_0_2Bdip, Beta_1_2Bdip;
    Q0 <<  -H_prll*std::pow(a,3)*(1-mu/mu0), -H_prll*std::pow(a,3)*(1-mu/mu0);
    Q1 <<  H_perp*std::pow(a,3)*(1-mu/mu0), H_perp*std::pow(a,3)*(1-mu/mu0);

    Beta_0_2Bdip=A0.colPivHouseholderQr().solve(Q0);
    Beta_1_2Bdip=A1.colPivHouseholderQr().solve(Q1);

    Beta_01_2Bdip=Beta_0_2Bdip(0);
    Beta_02_2Bdip=Beta_0_2Bdip(1);
    Beta_11_2Bdip=Beta_1_2Bdip(0);
    Beta_12_2Bdip=Beta_1_2Bdip(1);

    // correction force check
    Eigen::Vector3d Mi_2Bdip, Mj_2Bdip;

    Mi_2Bdip=Eigen::Vector3d::Zero();
    Mi_2Bdip= (Beta_01_2Bdip*z_cap + Beta_11_2Bdip*x_cap)*(4*M_PI*a*a*a);

    Mj_2Bdip=Eigen::Vector3d::Zero();
    Mj_2Bdip= (Beta_02_2Bdip*z_cap + Beta_12_2Bdip*x_cap)*(4*M_PI*a*a*a);

    double mir, mjr, mumu;
    mir=Mi_2Bdip.dot(SEP)/sep;
    mjr=Mj_2Bdip.dot(SEP)/sep;
    mumu = Mi_2Bdip.dot(Mj_2Bdip);

    double K = 3e-7/sep_sq/sep_sq;
    
    F_dip2B[0] = K*(mir*Mj_2Bdip[0]+mjr*Mi_2Bdip[0]+(mumu-5*mjr*mir)*SEP[0]/sep);
    F_dip2B[1] = K*(mir*Mj_2Bdip[1]+mjr*Mi_2Bdip[1]+(mumu-5*mjr*mir)*SEP[1]/sep);
    F_dip2B[2] = K*(mir*Mj_2Bdip[2]+mjr*Mi_2Bdip[2]+(mumu-5*mjr*mir)*SEP[2]/sep);

    Beta1_0[0]=Beta1_0[0] + Beta_01_dip - Beta_01_2Bdip;
    Beta2_0[0]=Beta2_0[0] + Beta_02_dip - Beta_02_2Bdip;
    Beta1_1[0]=Beta1_1[0] + Beta_11_dip - Beta_11_2Bdip;
    Beta2_1[0]=Beta2_1[0] + Beta_12_dip - Beta_12_2Bdip;

    // Create a 3D spherical mesh
    int N =180;
    double dang= M_PI/N;
    Eigen::VectorXd inc= Eigen::VectorXd::LinSpaced(N+1,dang/2, M_PI + dang/2).transpose();

    F=Eigen::Vector3d::Zero();
    
    // Integrating the force integrands
    for (int ii = 0; ii < N+1; ii++){
        double p;
        if (ii==0 or ii==N){
            p = 1;}
        else {p = 2;}
        double th=inc[ii];
        F[0]=F[0] + fx_int(th)*p*dang/2;
        F[2]=F[2] + fz_int(th)*p*dang/2;
    }

    F_act_coord=F[0]*x_cap + F[1]*y_cap + F[2]*z_cap;

    // std::cout<<"F"<<F.transpose()<<std::endl<<std::endl;
    // std::cout<<"F act coord"<<F_act_coord.transpose()<<std::endl<<std::endl;
    // Eigen::Matrix3d pre, post;
    // double th=1;
    // double ph=1;
    // pre<<   std::sin(th)*std::cos(ph), std::cos(th)*std::cos(ph), -std::sin(ph),
    //         std::sin(th)*std::sin(ph), std::cos(th)*std::sin(ph),  std::cos(ph),
    //         std::cos(th), -std::sin(th),  0;
    // post= pre.transpose();
    // Eigen::Vector3d H0_sph, H_sph, H_cart, rn_hat;
    // H0_sph=post*H0;
    // std::cout<<"mag_field"<<mag_field(a,th,ph).transpose() + H0_sph.transpose()<<std::endl<<std::endl;
    // std::cout<<"mag_field new "<<mag_A(a,th)*std::cos(ph)<<" "<<mag_B(a,th)*cos(ph)<<" "<<mag_C(a,th)*sin(ph)<<std::endl<<std::endl;
    

}

/////////////////////////////////////////
// Get functions to get force from the class
////////////////////////////////////////

Eigen::Vector3d spherical_harmonics::get_force(){
    return F;
}

Eigen::Vector3d spherical_harmonics::get_force_actual_coord(){
    return F_act_coord;
}

Eigen::Vector3d spherical_harmonics::get_force_2B_corrections(){
    return F_dip2B;
}

/////////////////////////////////////////
// Permutation function
////////////////////////////////////////

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

/////////////////////////////////////////
// FORCE INTEGRANDS
////////////////////////////////////////

double spherical_harmonics::fx_int(double theta){
    //-pi/4 (4 C P+3 C U+A W-(B (4 P+U)+A (4 Q+V)) Cos[theta]+(4 B Q-A (4 P+U)+B V+C W) Sin[theta]) 
    double term1, term2, term3, res;
    term1 = 4*mag_C(a,theta)*mag_P(a,theta)+3*mag_C(a,theta)*mag_U(a,theta)+mag_A(a,theta)*mag_W(a,theta);
    term2 = mag_B(a,theta)*(4*mag_P(a,theta)+mag_U(a,theta))+mag_A(a,theta)*(4*mag_Q(a,theta)+mag_V(a,theta));
    term3 = 4*mag_B(a,theta)*mag_Q(a,theta)-mag_A(a,theta)*(4*mag_P(a,theta)+mag_U(a,theta))+mag_B(a,theta)*mag_V(a,theta)+mag_C(a,theta)*mag_W(a,theta);

    res = -M_PI_4*(term1 - term2*std::cos(theta) + term3*std::sin(theta))*sin(theta)*mu0*a*a;
    return res;
}

double spherical_harmonics::fz_int(double theta){
    //pi/8 ((4 A*A-4 B*B-4 C*C+8 P*P-8 Q*Q+8 P U+3 U*U-8 Q V-3 V*V-W*W) Cos[theta]-2 (4 A B+8 P Q+4 Q U+4 P V+3 U V) Sin[theta]) 
    double mag_A_sq, mag_B_sq, mag_C_sq, mag_P_sq, mag_Q_sq, mag_U_sq, mag_V_sq, mag_W_sq;
    mag_A_sq=mag_A(a,theta)*mag_A(a,theta);
    mag_B_sq=mag_B(a,theta)*mag_B(a,theta);
    mag_C_sq=mag_C(a,theta)*mag_C(a,theta);
    mag_P_sq=mag_P(a,theta)*mag_P(a,theta);
    mag_Q_sq=mag_Q(a,theta)*mag_Q(a,theta);
    mag_U_sq=mag_U(a,theta)*mag_U(a,theta);
    mag_V_sq=mag_V(a,theta)*mag_V(a,theta);
    mag_W_sq=mag_W(a,theta)*mag_W(a,theta);
    //
    double mag_PU, mag_QV, mag_AB, mag_PQ, mag_QU, mag_PV, mag_UV;
    mag_PU=mag_P(a,theta)*mag_U(a,theta);
    mag_QV=mag_Q(a,theta)*mag_V(a,theta);
    mag_AB=mag_A(a,theta)*mag_B(a,theta);
    mag_PQ=mag_P(a,theta)*mag_Q(a,theta);
    mag_QU=mag_Q(a,theta)*mag_U(a,theta);
    mag_PV=mag_P(a,theta)*mag_V(a,theta);
    mag_UV=mag_U(a,theta)*mag_V(a,theta);
    //
    double term2, term3, res;
    term2 = 4*mag_A_sq-4*mag_B_sq-4*mag_C_sq+8*mag_P_sq-8*mag_Q_sq+8*mag_PU+3*mag_U_sq-8*mag_QV-3*mag_V_sq-mag_W_sq;
    term3 = 2*(4*mag_AB+8*mag_PQ+4*mag_QU+4*mag_PV+3*mag_UV);
    res = (M_PI/8)*(term2*std::cos(theta) - term3*std::sin(theta))*std::sin(theta)*mu0*a*a;
    return res;
}


/////////////////////////////////////////
// MAXWELL STRESS TENSOR FUNCTIONS
////////////////////////////////////////

double spherical_harmonics::mag_A(double r, double theta){
    double mag_A_res=0;
    for (int l = 1; l < L+1; l++){
        double Hrs1=0;
        for (int s = 1; s < L+1; s++){
            double Psm=lpmn_cos(1, s, theta);
            double ls_choose_sm=nchoosek(l+s, s+1);
            double r_pow_s_1=std::pow(r, (s-1));
            double sep_pow=std::pow(sep, l+s+1);
            double minus_one_pow=std::pow(-1, s+1);
            double r_pow_times_Psm=r_pow_s_1*Psm;
                
            double additional=(minus_one_pow)*s*r_pow_times_Psm/(sep_pow);
            Hrs1=Hrs1 + additional*(ls_choose_sm);
        }
        double Plm=lpmn_cos(1, l, theta);
        double r_pow_l2=std::pow(r, l+2);
        mag_A_res=mag_A_res + ((l+1)*Beta1_1[l-1]*(Plm/r_pow_l2) - Beta2_1[l-1]*Hrs1);
    }
    mag_A_res=mag_A_res + H_perp*std::sin(theta);
    return mag_A_res;
}

double spherical_harmonics::mag_P(double r, double theta){
    double mag_P_res=0;
    for (int l = 1; l < L+1; l++){
        double Hrs0=0;
        for (int s = 0; s < L+1; s++){
            double Psm=lpmn_cos(0, s, theta);
            double ls_choose_sm=nchoosek(l+s, s+0);
            double r_pow_s_1=std::pow(r, (s-1));
            double sep_pow=std::pow(sep, l+s+1);
            double minus_one_pow=std::pow(-1, s+0);
            double r_pow_times_Psm=r_pow_s_1*Psm;
                
            double additional=(minus_one_pow)*s*r_pow_times_Psm/(sep_pow);
            Hrs0=Hrs0 + additional*(ls_choose_sm);
        }
        double Plm=lpmn_cos(0, l, theta);
        double r_pow_l2=std::pow(r, l+2);
        mag_P_res=mag_P_res + ((l+1)*Beta1_0[l-1]*(Plm/r_pow_l2) - Beta2_0[l-1]*Hrs0);
    }
    mag_P_res= mag_P_res + H_prll*std::cos(theta);
    return mag_P_res;
}


double spherical_harmonics::mag_B(double r, double theta){
    double mag_B_res=0;
    for (int l = 1; l < L+1; l++){
        double Hths1=0;
        for (int s = 1; s < L+1; s++){
            double dPsm=d_lpmn_cos(1, s, theta);
            double ls_choose_sm=nchoosek(l+s, s+1);
            double r_pow_s_1=std::pow(r, (s-1));
            double sep_pow=std::pow(sep, l+s+1);
            double minus_one_pow=std::pow(-1, s+1);
            double r_pow_times_Psm=r_pow_s_1*dPsm;
                
            double additional=(minus_one_pow)*r_pow_times_Psm/(sep_pow);
            Hths1=Hths1 + additional*(ls_choose_sm);
        }
        double dPlm=d_lpmn_cos(1, l, theta);
        double r_pow_l2=std::pow(r, l+2);
        mag_B_res=mag_B_res - (Beta1_1[l-1]*(dPlm/r_pow_l2) + Beta2_1[l-1]*Hths1);
    }
    mag_B_res=mag_B_res + H_perp*std::cos(theta);
    return mag_B_res;
}


double spherical_harmonics::mag_Q(double r, double theta){
    double mag_Q_res=0;
    for (int l = 1; l < L+1; l++){
        double Hths0=0;
        for (int s = 0; s < L+1; s++){
            double dPsm=d_lpmn_cos(0, s, theta);
            double ls_choose_sm=nchoosek(l+s, s+0);
            double r_pow_s_1=std::pow(r, (s-1));
            double sep_pow=std::pow(sep, l+s+1);
            double minus_one_pow=std::pow(-1, s+0);
            double r_pow_times_Psm=r_pow_s_1*dPsm;
                
            double additional=(minus_one_pow)*r_pow_times_Psm/(sep_pow);
            Hths0=Hths0 + additional*(ls_choose_sm);
        }
        double dPlm=d_lpmn_cos(0, l, theta);
        double r_pow_l2=std::pow(r, l+2);
        mag_Q_res=mag_Q_res - (Beta1_0[l-1]*(dPlm/r_pow_l2) + Beta2_0[l-1]*Hths0);
    }
    mag_Q_res= mag_Q_res - H_prll*std::sin(theta);
    return mag_Q_res;
}


double spherical_harmonics::mag_C(double r, double theta){
    double mag_C_res=0;
    for (int l = 1; l < L+1; l++){
        double Hphs1=0;
        for (int s = 1; s < L+1; s++){
            double Psm=lpmn_cos(1, s, theta);
            double ls_choose_sm=nchoosek(l+s, s+1);
            double r_pow_s_1=std::pow(r, (s-1));
            double sep_pow=std::pow(sep, l+s+1);
            double minus_one_pow=std::pow(-1, s+1);
            double r_pow_times_Psm=r_pow_s_1*Psm;
                
            double additional=(minus_one_pow)*r_pow_times_Psm/(sep_pow);
            Hphs1=Hphs1 + additional*(ls_choose_sm)/std::sin(theta);
        }
        double Plm=lpmn_cos(1, l, theta);
        double r_pow_l2=std::pow(r, l+2);
        mag_C_res=mag_C_res + (Beta1_1[l-1]*(Plm/(r_pow_l2*std::sin(theta))) + Beta2_1[l-1]*Hphs1);
    }
    mag_C_res=mag_C_res - H_perp;
    return mag_C_res;
}

double spherical_harmonics::mag_U(double r, double theta){
    double mag_U_res;
    mag_U_res = (M_i.dot(y_cap)/(4*M_PI*a*a*a))*(lpmn_cos(1,1, theta)/(r*r))*std::sin(theta);
    return mag_U_res;
}

double spherical_harmonics::mag_V(double r, double theta){
    double mag_V_res;
    mag_V_res = (M_i.dot(y_cap)/(4*M_PI*a*a*a))*(lpmn_cos(1,1, theta)/(r*r))*std::cos(theta);
    return mag_V_res;
}

double spherical_harmonics::mag_W(double r, double theta){
    double mag_W_res;
    mag_W_res = (M_i.dot(y_cap)/(4*M_PI*a*a*a))*(lpmn_cos(1,1, theta)/(r*r));
    return mag_W_res;
}

/////////////////////////////////////////////////////////
//define associate legendre functions for cos(theta)
/////////////////////////////////////////////////////////

double spherical_harmonics::lpmn_cos(int m, int n, double theta){
    return boost::math::legendre_p(n, m, std::cos(theta));
}

double spherical_harmonics::d_lpmn_cos(int m, int n, double theta){
    return ((m-n-1)*lpmn_cos(m,n+1,theta) + (n+1)*std::cos(theta)*lpmn_cos(m,n,theta))/(-std::sin(theta));
}