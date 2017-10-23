#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF()
{
    is_initialized_ = false;

    previous_timestamp_ = 0;

    // initializing matrices
    R_laser_ = MatrixXd(2, 2);
    R_radar_ = MatrixXd(3, 3);
    H_laser_ = MatrixXd(2, 4);
    Hj_ = MatrixXd(3, 4);

    //measurement covariance matrix - laser
    R_laser_ << 0.0225, 0,
        0, 0.0225;

    //measurement covariance matrix - radar
    R_radar_ << 0.09, 0, 0,
        0, 0.0009, 0,
        0, 0, 0.09;
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF()
{}

void FusionEKF::ProcessMeasurement(const MeasurementPackage& measurement_pack)
{


    /*****************************************************************************
     *  Initialization
     ****************************************************************************/
    if (!is_initialized_)
    {
        // first measurement
        cout << "EKF: " << endl;

        //create a 4D state vector, we don't know yet the values of the x state
        ekf_.x_ = VectorXd(4);

        //state covariance matrix P
        ekf_.P_ = MatrixXd(4, 4);
        ekf_.P_ << 100, 0, 0, 0,
            0, 100, 0, 0,
            0, 0, 1000, 0,
            0, 0, 0, 1000;


        //measurement covariance
        ekf_.R_ = MatrixXd(2, 2);
        ekf_.R_ << 0.0225, 0,
            0, 0.0225;

        //measurement matrix
        ekf_.H_ = MatrixXd(2, 4);
        ekf_.H_ << 1, 0, 0, 0,
            0, 1, 0, 0;

        //the initial transition matrix F_
        ekf_.F_ = MatrixXd(4, 4);
        ekf_.F_ << 1, 0, 1, 0,
            0, 1, 0, 1,
            0, 0, 1, 0,
            0, 0, 0, 1;

        ekf_.Q_ = MatrixXd(4, 4);
        ekf_.Q_ << 0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0;

        //set the acceleration noise components
        noise_ax_ = 9;
        noise_ay_ = 9;

        if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR)
        {
            /**
            Convert radar from polar to cartesian coordinates and initialize state.
            */
            double r = measurement_pack.raw_measurements_(0);
            double phi = measurement_pack.raw_measurements_(1);
            ekf_.x_ << r * cos(phi), r * sin(phi), 0, 0;
        }
        else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER)
        {
            ekf_.x_ << measurement_pack.raw_measurements_(
                0), measurement_pack.raw_measurements_(1), 0, 0;
        }

        // done initializing, no need to predict or update
        is_initialized_ = true;
        previous_timestamp_ = measurement_pack.timestamp_;
        return;
    }

    /*****************************************************************************
     *  Prediction
     ****************************************************************************/

    //compute the time elapsed between the current and previous measurements
    double dt = (measurement_pack.timestamp_ - previous_timestamp_) /
                1000000.0;    //dt - expressed in seconds
    previous_timestamp_ = measurement_pack.timestamp_;
    //1. Modify the F matrix so that the time is integrated
    ekf_.F_ << 1, 0, dt, 0,
        0, 1, 0, dt,
        0, 0, 1, 0,
        0, 0, 0, 1;

    //2. Set the process covariance matrix Q
    double dt2 = dt * dt;
    double dt3 = dt * dt2;
    double dt4 = dt * dt3;
    ekf_.Q_ = MatrixXd(4, 4);
    ekf_.Q_ << noise_ax_ * dt4 / 4.0, 0, noise_ax_ * dt3 / 2.0, 0,
        0, noise_ay_ * dt4 / 4.0, 0, noise_ay_ * dt3 / 2.0,
        noise_ax_ * dt3 / 2.0, 0, noise_ax_ * dt2, 0,
        0, noise_ay_ * dt3 / 2.0, 0, noise_ay_ * dt2;


    ekf_.Predict();

    /*****************************************************************************
     *  Update
     ****************************************************************************/

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR)
    {
        ekf_.R_ = R_radar_;
        ekf_.UpdateEKF(measurement_pack.raw_measurements_);
    }
    else
    {
        ekf_.R_ = R_laser_;
        ekf_.Update(measurement_pack.raw_measurements_);
    }

    // print the output
    cout << "x_ = " << ekf_.x_ << endl;
    cout << "P_ = " << ekf_.P_ << endl;
}
