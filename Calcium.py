#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 12:57:38 2021
@author: Mathew
"""

from skimage.io import imread
import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import filters, measure
from skimage.filters import threshold_local


# Example change

# Filename to look for
filename_contains = ".tif"

# Where to save the results
root_path = "/Volumes/T7/Current_Analysis/JL_CaExperiments/Ca3/"

pathlist = []

pathlist.append("/Volumes/T7/Current_Analysis/JL_CaExperiments/Ca3/1/")


# Threshold to use:

thresh=500

def load_image(toload):
    return imread(toload)


def z_project(image):
    return np.mean(image, axis=0)


# Subtract background
def subtract_bg(image):
    background = threshold_local(image, 11, offset=np.percentile(image, 1), method='median')
    return image - background


def threshold_image_otsu(input_image):
    threshold_value = filters.threshold_otsu(input_image)
    print(threshold_value)
    binary_image = input_image > threshold_value
    return threshold_value, binary_image


def threshold_image_standard(input_image, thresh):
    binary_image = input_image > thresh
    return binary_image


def threshold_image_fixed(input_image, threshold_number):
    threshold_value = threshold_number
    binary_image = input_image > threshold_value
    return threshold_value, binary_image


def label_image(input_image):
    labelled_image = measure.label(input_image)
    number_of_features = labelled_image.max()
    return number_of_features, labelled_image


def show(input_image, color=''):
    cmap = {"Red": "Reds", "Blue": "Blues", "Green": "Greens"}.get(color, None)
    plt.imshow(input_image, cmap=cmap)
    plt.show()


def create_new_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def analyse_labelled_image(labelled_image, original_image):
    measure_image = measure.regionprops_table(
        labelled_image, intensity_image=original_image,
        properties=('area', 'perimeter', 'centroid', 'orientation',
                    'major_axis_length', 'minor_axis_length', 'mean_intensity', 'max_intensity')
    )
    return pd.DataFrame.from_dict(measure_image)


def save_image(image, path, filename):
    im = Image.fromarray(image)
    im.save(os.path.join(path, filename))


def save_plot(data, path, filename, bins, rng, xlabel, ylabel, title=''):
    plt.hist(data, bins=bins, range=rng, rwidth=0.9, color='#ff0000')
    plt.xlabel(xlabel, size=20)
    plt.ylabel(ylabel, size=20)
    if title:
        plt.title(title, size=20)
    plt.savefig(os.path.join(path, filename))
    plt.show()


def get_statistics(measurements, num_features):
    stats = {
        'num_features': num_features,
        'mean_area': measurements['area'].mean(),
        'std_area': measurements['area'].std(),
        'mean_perimeter': measurements['perimeter'].mean(),
        'std_perimeter': measurements['perimeter'].std(),
        'mean_major_axis_length': measurements['major_axis_length'].mean(),
        'std_major_axis_length': measurements['major_axis_length'].std(),
        'mean_minor_axis_length': measurements['minor_axis_length'].mean(),
        'std_minor_axis_length': measurements['minor_axis_length'].std(),
        'mean_mean_intensity': measurements['mean_intensity'].mean(),
        'std_mean_intensity': measurements['mean_intensity'].std(),
        'mean_max_intensity': measurements['max_intensity'].mean(),
        'std_max_intensity': measurements['max_intensity'].std()
    }
    return stats


# Initialize an empty DataFrame to store all statistics
all_statistics = pd.DataFrame()

for path in pathlist:
    j = 0
    for root, dirs, files in os.walk(path):
        for name in files:
            if filename_contains in name and 'out.tif' not in name:
              if '._' not in name:
                image_path = os.path.join(path, name)
                print(image_path)
                new_path = os.path.join(path, str(j))
                create_new_directory(new_path)
                j += 1
                
                # Load image
                stack = load_image(image_path)
                
                # Extract channels
                calcium = stack[1:3, :, :, 0]
                mito = stack[1:3, :, :, 1]
                dapi = stack[1:3, :, :, 2]
                
                # Z-project (average)
                calcium_flat = z_project(calcium)
                save_image(calcium_flat, new_path, 'calcium_out.tif')
                
                mito_flat = z_project(mito)
                save_image(mito_flat, new_path, 'mito_out.tif')
                
                dapi_flat = z_project(dapi)
                save_image(dapi_flat, new_path, 'dapi_out.tif')
                
                # Threshold image
                binary = threshold_image_standard(calcium_flat, thresh)
                save_image(binary, new_path, 'calcium_binary_out.tif')
                
                # Get total intensity of all pixels
                thresholded=binary*calcium_flat
                
                # Label the image
                number, labelled = label_image(binary)
                save_image(labelled, new_path, 'calcium_labelled_out.tif')
                
                # Analyse the image
                measurements = analyse_labelled_image(labelled, calcium_flat)
                
                measurements.to_csv(new_path + '/Metrics.csv', sep = '\t')
                
                # Collect statistics including the number of features
                stats = get_statistics(measurements, number)
                stats['image_name'] = name
                stats['image_path'] = image_path
                stats['number_of_pixels_with_calcium']=binary.sum()
                stats['total_intensity_of_calcium_pixels']=thresholded.sum()
                all_statistics = pd.concat([all_statistics, pd.DataFrame([stats])], ignore_index=True)
                
                intensity = measurements['mean_intensity']
                save_plot(intensity, new_path, "intensity_hist.pdf", bins=20, rng=[0, 10000], xlabel='Intensity', ylabel='Number of Features')
                
                areas = measurements['area']
                save_plot(areas, new_path, "area_hist.pdf", bins=30, rng=[0, 300], xlabel='Area (pixels)', ylabel='Number of Features')
                
                length = measurements['major_axis_length']
                save_plot(length, new_path, "Lengths.pdf", bins=20, rng=[0, 100], xlabel='Length', ylabel='Number of Features', title='Cluster lengths')
                

# Save the collected statistics to a CSV file
output_csv_path = os.path.join(root_path, 'all_image_statistics.csv')
all_statistics.to_csv(output_csv_path, index=False)
print(f"Collected statistics saved to {output_csv_path}")
