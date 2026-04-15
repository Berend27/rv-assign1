#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"

#include <iostream>
#include <string>

using namespace cv;
using namespace std;

static void print_help(char** argv)
{
    cout << "Usage:\n"
         << argv[0]
         << " <left_image> <right_image> <intrinsics_yml> <extrinsics_yml> <rectified_left_out> <rectified_right_out>\n";
}

int main(int argc, char** argv)
{
    if (argc != 7)
    {
        print_help(argv);
        return 1;
    }

    string leftImagePath = argv[1];
    string rightImagePath = argv[2];
    string intrinsicsPath = argv[3];
    string extrinsicsPath = argv[4];
    string rectifiedLeftOut = argv[5];
    string rectifiedRightOut = argv[6];

    Mat imgLeft = imread(leftImagePath, IMREAD_COLOR);
    Mat imgRight = imread(rightImagePath, IMREAD_COLOR);

    if (imgLeft.empty())
    {
        cerr << "Error: could not load left image: " << leftImagePath << endl;
        return 1;
    }
    if (imgRight.empty())
    {
        cerr << "Error: could not load right image: " << rightImagePath << endl;
        return 1;
    }
    if (imgLeft.size() != imgRight.size())
    {
        cerr << "Error: left and right images have different sizes." << endl;
        return 1;
    }

    FileStorage fsIntr(intrinsicsPath, FileStorage::READ);
    if (!fsIntr.isOpened())
    {
        cerr << "Error: could not open intrinsics file: " << intrinsicsPath << endl;
        return 1;
    }

    Mat M1, D1, M2, D2;
    fsIntr["M1"] >> M1;
    fsIntr["D1"] >> D1;
    fsIntr["M2"] >> M2;
    fsIntr["D2"] >> D2;
    fsIntr.release();

    if (M1.empty() || D1.empty() || M2.empty() || D2.empty())
    {
        cerr << "Error: intrinsics file is missing M1/D1/M2/D2." << endl;
        return 1;
    }

    FileStorage fsExtr(extrinsicsPath, FileStorage::READ);
    if (!fsExtr.isOpened())
    {
        cerr << "Error: could not open extrinsics file: " << extrinsicsPath << endl;
        return 1;
    }

    Mat R, T;
    fsExtr["R"] >> R;
    fsExtr["T"] >> T;
    fsExtr.release();

    if (R.empty() || T.empty())
    {
        cerr << "Error: extrinsics file is missing R or T." << endl;
        return 1;
    }

    Size imageSize = imgLeft.size();

    Mat R1, R2, P1, P2, Q;
    Rect roi1, roi2;

    stereoRectify(
        M1, D1, M2, D2, imageSize, R, T,
        R1, R2, P1, P2, Q,
        CALIB_ZERO_DISPARITY, -1, imageSize, &roi1, &roi2
    );

    Mat map11, map12, map21, map22;
    initUndistortRectifyMap(M1, D1, R1, P1, imageSize, CV_16SC2, map11, map12);
    initUndistortRectifyMap(M2, D2, R2, P2, imageSize, CV_16SC2, map21, map22);

    Mat rectLeft, rectRight;
    remap(imgLeft, rectLeft, map11, map12, INTER_LINEAR);
    remap(imgRight, rectRight, map21, map22, INTER_LINEAR);

    if (!imwrite(rectifiedLeftOut, rectLeft))
    {
        cerr << "Error: failed to write rectified left image: " << rectifiedLeftOut << endl;
        return 1;
    }
    if (!imwrite(rectifiedRightOut, rectRight))
    {
        cerr << "Error: failed to write rectified right image: " << rectifiedRightOut << endl;
        return 1;
    }

    cout << "Rectified left image written to: " << rectifiedLeftOut << endl;
    cout << "Rectified right image written to: " << rectifiedRightOut << endl;
    cout << "Q matrix:" << endl << Q << endl;

    return 0;
}