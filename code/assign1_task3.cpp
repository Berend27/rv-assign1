#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <limits>

using namespace cv;
using namespace std;

static void print_help(char** argv)
{
    cout << "Usage:\n"
         << argv[0]
         << " <left_image> <right_image> <intrinsics_yml> <extrinsics_yml> "
         << " <rectified_left_out> <rectified_right_out> <disparity_pfm> <ply_out>\n";
}

static bool loadPFM(const string& filename, Mat& img)
{
    ifstream file(filename, ios::binary);
    if (!file.is_open())
    {
        cerr << "Error: could not open PFM file: " << filename << endl;
        return false;
    }

    string type;
    file >> type;
    if (type != "Pf" && type != "PF")
    {
        cerr << "Error: unsupported PFM type in " << filename << ": " << type << endl;
        return false;
    }

    int width = 0, height = 0;
    file >> width >> height;

    float scale = 0.0f;
    file >> scale;
    file.get(); // consume single whitespace/newline after header

    if (width <= 0 || height <= 0)
    {
        cerr << "Error: invalid PFM dimensions in " << filename << endl;
        return false;
    }

    bool color = (type == "PF");
    int channels = color ? 3 : 1;
    vector<float> data(static_cast<size_t>(width) * height * channels);

    file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(float)));
    if (!file)
    {
        cerr << "Error: failed reading PFM pixel data from " << filename << endl;
        return false;
    }

    int matType = color ? CV_32FC3 : CV_32FC1;
    Mat tmp(height, width, matType, data.data());
    tmp = tmp.clone();

    // Negative scale means little-endian, which is what we expect on macOS.
    // PFM stores rows from bottom to top, so flip vertically.
    flip(tmp, img, 0);

    return true;
}

static bool writeColorPLY(const string& filename, const Mat& xyz, const Mat& color, float maxZ = 10000.0f)
{
    if (xyz.empty() || color.empty())
    {
        cerr << "Error: empty xyz or color image for PLY export." << endl;
        return false;
    }
    if (xyz.size() != color.size())
    {
        cerr << "Error: xyz and color image sizes do not match." << endl;
        return false;
    }
    if (xyz.type() != CV_32FC3)
    {
        cerr << "Error: xyz matrix must be CV_32FC3." << endl;
        return false;
    }
    if (color.type() != CV_8UC3)
    {
        cerr << "Error: color image must be CV_8UC3." << endl;
        return false;
    }

    size_t count = 0;
    for (int y = 0; y < xyz.rows; ++y)
    {
        for (int x = 0; x < xyz.cols; ++x)
        {
            Vec3f p = xyz.at<Vec3f>(y, x);
            if (!isfinite(p[0]) || !isfinite(p[1]) || !isfinite(p[2]))
                continue;
            if (fabs(p[2]) > maxZ || p[2] <= 0.0f)
                continue;
            count++;
        }
    }

    ofstream out(filename);
    if (!out.is_open())
    {
        cerr << "Error: could not open output PLY file: " << filename << endl;
        return false;
    }

    out << "ply\n";
    out << "format ascii 1.0\n";
    out << "element vertex " << count << "\n";
    out << "property float x\n";
    out << "property float y\n";
    out << "property float z\n";
    out << "property uchar red\n";
    out << "property uchar green\n";
    out << "property uchar blue\n";
    out << "end_header\n";

    for (int y = 0; y < xyz.rows; ++y)
    {
        for (int x = 0; x < xyz.cols; ++x)
        {
            Vec3f p = xyz.at<Vec3f>(y, x);
            if (!isfinite(p[0]) || !isfinite(p[1]) || !isfinite(p[2]))
                continue;
            if (fabs(p[2]) > maxZ || p[2] <= 0.0f)
                continue;

            Vec3b bgr = color.at<Vec3b>(y, x);
            int r = bgr[2];
            int g = bgr[1];
            int b = bgr[0];

            out << p[0] << " " << -p[1] << " " << p[2] << " "
                << r << " " << g << " " << b << "\n";
        }
    }

    return true;
}

int main(int argc, char** argv)
{
    if (argc != 9)
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
    string disparityPFMPath = argv[7];
    string plyOut = argv[8];

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

    Mat disparity;
    if (!loadPFM(disparityPFMPath, disparity))
    {
        return 1;
    }

    if (disparity.type() != CV_32FC1)
    {
        cerr << "Error: expected single-channel float disparity in PFM file." << endl;
        return 1;
    }

    if (disparity.size() != imageSize)
    {
        cerr << "Error: disparity size does not match image size." << endl;
        cerr << "Disparity size: " << disparity.cols << " x " << disparity.rows << endl;
        cerr << "Image size: " << imageSize.width << " x " << imageSize.height << endl;
        return 1;
    }

    Mat xyz;
    reprojectImageTo3D(disparity, xyz, Q, true);

    if (!writeColorPLY(plyOut, xyz, rectLeft))
    {
        return 1;
    }

    cout << "Rectified left image written to: " << rectifiedLeftOut << endl;
    cout << "Rectified right image written to: " << rectifiedRightOut << endl;
    cout << "Loaded disparity PFM: " << disparityPFMPath << endl;
    cout << "PLY written to: " << plyOut << endl;
    cout << "Q matrix:" << endl << Q << endl;

    return 0;
}