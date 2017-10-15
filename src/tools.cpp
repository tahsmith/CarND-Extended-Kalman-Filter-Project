#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools()
{}

Tools::~Tools()
{}

VectorXd Tools::CalculateRMSE(const vector<VectorXd>& estimations,
                              const vector<VectorXd>& ground_truth)
{
    VectorXd rmse(4);
    rmse << 0, 0, 0, 0;
    size_t count = 0;
    for (int i = 0; i < estimations.size(); ++i)
    {
        VectorXd err = estimations[i] - ground_truth[i];
        if ((estimations[i].size() != ground_truth[i].size())
            || estimations[i].size() == 0)
        {
            continue;
        }
        VectorXd err_sq = err.array() * err.array();
        rmse += err_sq;
        count += 1;
    }

    //calculate the mean
    rmse = rmse / count;

    //calculate the squared root
    rmse = rmse.array().sqrt();

    //return the result
    return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state)
{
    MatrixXd Hj(3, 4);
    //recover state parameters
    double px = x_state(0);
    double py = x_state(1);
    double vx = x_state(2);
    double vy = x_state(3);

    double rp2 = px * px + py * py;
    if (rp2 == 0)
    {
        std::cout << "warning: cannot evaluate jacobian at: (" << px << ", "
                  << py << ")." << std::endl;
        return Hj;
    }
    float rp = sqrt(rp2);

    Hj <<
       px / rp, py / rp, 0, 0,
        -py / rp2, px / rp2, 0, 0,
        py * (vx * py - vy * px) / rp / rp2, px * (vy * px - vx * py) / rp /
                                             rp2, px / rp, py / rp;

    return Hj;
}
