// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// Copyright (c) 2012, 2013, 2015 Pierre MOULON.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "openMVG/cameras/cameras.hpp"
#include "openMVG/geodesy/geodesy.hpp"
#include "openMVG/image/image_io.hpp"
#include "openMVG/numeric/eigen_alias_definition.hpp"
#include "openMVG/sfm/sfm_data.hpp"
#include "openMVG/sfm/sfm_data_io.hpp"
#include "openMVG/sfm/sfm_data_utils.hpp"
#include "openMVG/sfm/sfm_view.hpp"
#include "openMVG/sfm/sfm_view_priors.hpp"
#include "openMVG/types.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <utility>

using namespace openMVG;
using namespace openMVG::cameras;
using namespace openMVG::geodesy;
using namespace openMVG::image;
using namespace openMVG::sfm;



class OpenMVG_PipeLine
{
    std::vector<cv::Mat> img;
    std::vector<cv::Mat> descriptor;
    std::vector<std::vector< cv::KeyPoint >> imageKeyPoints;
    std::string imageDir;
    std::string maskDir;
    SfM_Data sfm_data;
    std::vector<cv::String> imageName; // Image name
    EINTRINSIC e_User_camera_model;
    int i_User_camera_model;
    double focal_pixels;
    std::ostringstream error_report_stream;

public :
    OpenMVG_PipeLine(cv::String folder, cv::String mask);

    int getFolderImage();
    void setUserCameraModel(EINTRINSIC v);
    void setFocalPixels(double  f);
    bool Error();
    void printError();
    void ComputeDescriptor();


};

OpenMVG_PipeLine::OpenMVG_PipeLine(cv::String folder, cv::String mask)
{
    imageDir = folder;
    maskDir = mask;
    sfm_data.s_root_path = imageDir;
}

int OpenMVG_PipeLine::getFolderImage()
{
    std::string sPriorWeights;
    std::pair<bool, Vec3> prior_w_info(false, Vec3(1.0, 1.0, 1.0));
    bool b_Group_camera_model = true;
    // Expected properties for each image
    double width = -1, height = -1, focal = -1, ppx = -1, ppy = -1;

    cv::glob(imageDir+maskDir, imageName);

    // Configure an empty scene with Views and their corresponding cameras
    Views & views = sfm_data.views;
    Intrinsics & intrinsics = sfm_data.intrinsics;

    std::cout << "\n- Image listing -\n";
    std::ostringstream error_report_stream;
    for (std::vector<cv::String>::const_iterator iter_image = imageName.begin();
        iter_image != imageName.end();
        ++iter_image)
    {
        // Read meta data to fill camera parameter (w,h,focal,ppx,ppy) fields.
        width = height = ppx = ppy = focal = -1.0;

        cv::Mat x = cv::imread(*iter_image);

        ImageHeader imgHeader;
        if (x.empty())
            continue; // image cannot be read
        img.push_back(x);
        width = x.cols;
        height = x.rows;
        ppx = width / 2.0;
        ppy = height / 2.0;
        focal = focal_pixels;

        // Build intrinsic parameter related to the view
        std::shared_ptr<IntrinsicBase> intrinsic;

        if (focal > 0 && ppx > 0 && ppy > 0 && width > 0 && height > 0)
        {
            // Create the desired camera type
            switch (e_User_camera_model)
            {
            case PINHOLE_CAMERA:
                intrinsic = std::make_shared<Pinhole_Intrinsic>
                    (width, height, focal, ppx, ppy);
                break;
            case PINHOLE_CAMERA_RADIAL1:
                intrinsic = std::make_shared<Pinhole_Intrinsic_Radial_K1>
                    (width, height, focal, ppx, ppy, 0.0); // setup no distortion as initial guess
                break;
            case PINHOLE_CAMERA_RADIAL3:
                intrinsic = std::make_shared<Pinhole_Intrinsic_Radial_K3>
                    (width, height, focal, ppx, ppy, 0.0, 0.0, 0.0);  // setup no distortion as initial guess
                break;
            case PINHOLE_CAMERA_BROWN:
                intrinsic = std::make_shared<Pinhole_Intrinsic_Brown_T2>
                    (width, height, focal, ppx, ppy, 0.0, 0.0, 0.0, 0.0, 0.0); // setup no distortion as initial guess
                break;
            case PINHOLE_CAMERA_FISHEYE:
                intrinsic = std::make_shared<Pinhole_Intrinsic_Fisheye>
                    (width, height, focal, ppx, ppy, 0.0, 0.0, 0.0, 0.0); // setup no distortion as initial guess
                break;
            default:
                std::cerr << "Error: unknown camera model: " << (int)e_User_camera_model << std::endl;
                return EXIT_FAILURE;
            }
        }

        View v(*iter_image, views.size(), views.size(), views.size(), width, height);

        // Add intrinsic related to the image (if any)
        if (intrinsic == nullptr)
        {
            //Since the view have invalid intrinsic data
            // (export the view, with an invalid intrinsic field value)
            v.id_intrinsic = UndefinedIndexT;
        }
        else
        {
            // Add the defined intrinsic to the sfm_container
            intrinsics[v.id_intrinsic] = intrinsic;
        }

        // Add the view to the sfm_container
        views[v.id_view] = std::make_shared<View>(v);
    }
    // Group camera that share common properties if desired (leads to more faster & stable BA).
    if (b_Group_camera_model)
    {
        GroupSharedIntrinsics(sfm_data);
    }
    return img.size();
}

void OpenMVG_PipeLine::setUserCameraModel(EINTRINSIC v)
{
    e_User_camera_model = EINTRINSIC(v);

}

void OpenMVG_PipeLine::setFocalPixels(double f)
{
    focal_pixels = f;
}

bool OpenMVG_PipeLine::Error()
{
    return error_report_stream.str().empty();
}

void OpenMVG_PipeLine::printError()
{
    std::cerr
        << "\nWarning & Error messages:" << std::endl
        << error_report_stream.str() << std::endl;

}

void OpenMVG_PipeLine::ComputeDescriptor()
{
    cv::Ptr<cv::Feature2D> detector = cv::xfeatures2d::SIFT::create(1000);

    for (int i= imageKeyPoints.size();i<img.size();i++)
    {
        std::vector< cv::KeyPoint > keyPoints;
        cv::Mat desc;
        detector->detectAndCompute(img[i],cv::Mat(), keyPoints,desc);
        imageKeyPoints.push_back(keyPoints);
        descriptor.push_back(desc);
    }

}

//
// Create the description of an input image dataset for OpenMVG toolsuite
// - Export a SfM_Data file with View & Intrinsic data
//
int main(int argc, char **argv)
{
    OpenMVG_PipeLine myExample("G:\\Lib\\build\\openMVG\\Images\\", "*.png");


    myExample.setFocalPixels(1426.0);
    myExample.setUserCameraModel(EINTRINSIC::PINHOLE_CAMERA_RADIAL3);
    myExample.getFolderImage();
    myExample.ComputeDescriptor();
    // Display saved warning & error messages if any.
    if (!myExample.Error())
    {
            myExample.printError();
    }



    return EXIT_SUCCESS;
}
