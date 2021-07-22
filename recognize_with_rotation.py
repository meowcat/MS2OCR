# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 11:05:48 2020

@author: stravsm
"""

import imageio
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
import pandas as pd

from deskew import determine_skew
import pytesseract
from io import StringIO
import re
from scipy import stats
from sklearn.linear_model import RANSACRegressor

from tqdm import tqdm
import csv

f = r"C:\Daten\MetIDifyer\nigriventine\Untitled.png"

src = cv.imread(f, cv.IMREAD_COLOR)

# Get vertical lines:
# mostly from https://stackoverflow.com/questions/53831171/get-line-coordinates-for-vertical-lines-numpy


if len(src.shape) != 2:
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
else:
    gray = src
   
gray = cv.bitwise_not(gray)
bw = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, -2)
vertical = np.copy(bw)

img = src.copy()
rows = vertical.shape[0]
verticalsize = int(rows / 30)
verticalStructure = cv.getStructuringElement(cv.MORPH_RECT, (1, verticalsize))
vertical = cv.erode(vertical, verticalStructure)
vertical = cv.dilate(vertical, verticalStructure)
contours = cv.findContours(vertical, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]

#detected_vertical = cv.morphologyEx(bw, cv.MORPH_OPEN, verticalStructure, iterations=2)

src_delined = src.copy()
for c in contours:
    cv.drawContours(src_delined, [c], -1, (255,255,255), 4)

plt.imshow(vertical)
v_transpose = np.transpose(np.nonzero(vertical))

minLineLength = 5
maxLineGap = 10
threshold = 5
diff_threshold = 6

lines = cv.HoughLinesP(vertical,1,np.pi/180,threshold,minLineLength,maxLineGap)
for line in lines:
    for x1,y1,x2,y2 in line:
        cv.line(img,(x1,y1),(x2,y2),(0,255,0),2)

lines_df = pd.DataFrame(lines.squeeze(1))
lines_df.columns = ["x1", "yr1", "x2", "yr2"]
lines_df["y2"] = np.minimum(lines_df["yr1"], lines_df["yr2"])
lines_df["y1"] = np.maximum(lines_df["yr1"], lines_df["yr2"])
lines_df["intensity_abs"] = lines_df["y1"] - lines_df["y2"]
lines_df.sort_values("x1", inplace=True)
lines_df["diff"] = lines_df["x1"].diff().fillna(np.Inf)
lines_df = lines_df.loc[lines_df["diff"] > diff_threshold]
lines_df.reset_index(inplace=True)
plt.imshow(img)

# grayscale = cv.cvtColor(src_delined, cv.COLOR_BGR2GRAY)
# determine_skew(grayscale, sigma=3,   num_peaks = 10)

# Get peak annotations with OCR
# Rotation:
# https://gist.github.com/jarodsmk/4d3c0f19fba9c386cfec292513e946b4
# https://stackoverflow.com/questions/22041699/rotate-an-image-without-cropping-in-opencv-in-c

# osd_info = pytesseract.image_to_osd(src)
# rotation = float(re.search("Rotate: ([0-9]+)\n", osd_info).group(1))
# if rotation > 0:
#     rotation = 360 - rotation

def rotate_image(mat, angle):
  # angle in degrees

  height, width = mat.shape[:2]
  image_center = (width/2, height/2)

  rotation_mat = cv.getRotationMatrix2D(image_center, angle, 1.)
  abs_cos = abs(rotation_mat[0,0])
  abs_sin = abs(rotation_mat[0,1])

  bound_w = int(height * abs_sin + width * abs_cos)
  bound_h = int(height * abs_cos + width * abs_sin)

  rotation_mat[0, 2] += bound_w/2 - image_center[0]
  rotation_mat[1, 2] += bound_h/2 - image_center[1]

  rotated_mat = cv.warpAffine(mat, rotation_mat, (bound_w, bound_h))
  return rotated_mat, rotation_mat



def score_rotation(img, rotation):
    rotated, _ = rotate_image(img, rotation)
    ocr = pytesseract.image_to_string(rotated, config = '--psm 11')
    score = len(re.findall('[0-9]', ocr))
    return score


# Find the best rotation:
# Rotation at which most digits are recognized
# Ignore the dot because strange things might look like dots
# First find the rough value with 2-degree steps,
# then find the best value around the max in 0.5 degree steps
rotation_seq = range(0,360,2)

rotation_score = [score_rotation(src, x) for x in tqdm(rotation_seq)]
plt.plot(rotation_seq,rotation_score)

rotation_center = rotation_seq[np.argmax(rotation_score)]
rotation_fine = [rotation_center + x for x in np.arange(-5, 5, 0.5)]

rotation_score_fine = [score_rotation(src, x) for x in tqdm(rotation_fine)]
rotation = rotation_fine[np.argmax(rotation_score_fine)]
plt.plot(rotation_fine, rotation_score_fine)

rotation = 0

rotated, M = rotate_image(src, rotation)
im_data_ = pytesseract.image_to_data(rotated)

M_inv = cv.invertAffineTransform(M)


pytesseract.image_to_string(rotated, config = '--psm 11')
pytesseract.image_to_string(rotated)


# (h, w) = src.shape[:2]
# max_size = max(h, w)
# center = (w // 2, h // 2)
# M = cv.getRotationMatrix2D(center, rotation, 1.0)
# rotated = cv.warpAffine(src, M, (2*max_size, 2*max_size),
#  	flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)	
plt.imshow(rotated)

#pytesseract.image_to_string(rotated)
im_data_ = pytesseract.image_to_data(rotated, config = '--psm 11')
im_data = pd.read_table(StringIO(im_data_), sep='\t', quoting=csv.QUOTE_NONE)
im_data = im_data.loc[im_data["text"].notna()]
im_data=  im_data.loc[im_data["text"].str.match("([0-9][0-9.]+)")]
im_data = im_data.copy()
im_data["mz_annotated"] = pd.to_numeric(
    im_data["text"].str.replace("[^0-9.]",""))

im_data.drop(["page_num", "block_num", "par_num",
              "line_num", "word_num", "text"], 
             axis=1, inplace=True)

# Transform points back
transform_inv = lambda points, M_inv: \
    pd.DataFrame(cv.transform(points.to_numpy().
                 astype('float32').
                 reshape(-1,1,2), M_inv).reshape(-1, 2)).\
    rename({0: 'x', 1: 'y'}, axis=1)

im_data["right"] = im_data["left"] + im_data["width"]
im_data["bottom"] = im_data["top"] + im_data["height"]
im_data.reset_index(inplace = True)
im_data.drop("index", axis=1, inplace=True)


col_tf = transform_inv(im_data[["left", "top"]], M_inv)
im_data[["left", "top"]] = col_tf
col_tf = transform_inv(im_data[["right", "bottom"]], M_inv)
im_data[["right", "bottom"]] = col_tf
im_data["center"] = (im_data["left"] +im_data["right"]) / 2.



# Align: find unambiguous matches and calibrate
tol_pixel_initial = 10

def match_line(x, xref, tol):
    matching = np.abs(xref - x) < tol
    if sum(matching) == 1:
        return np.argmax(matching)
    return np.nan

im_data["lines_index"] = im_data["center"].map(lambda x:
                                               match_line(x, 
                                               xref = lines_df["x1"],
                                               tol = tol_pixel_initial))
im_data["lines_match"] = im_data["lines_index"]
spectrum_data_cal = pd.merge(im_data, lines_df, how="outer", left_on='lines_index', right_index=True)
spectrum_subset = spectrum_data_cal.loc[spectrum_data_cal["lines_match"].notna()]
# mz_slope, mz_intercept, _, _ = stats.theilslopes(spectrum_subset["x1"], spectrum_subset["mz_annotated"])
mz_slope, mz_intercept, _, _, _ = stats.linregress(spectrum_subset["x1"], spectrum_subset["mz_annotated"])
mz_slope, mz_intercept, _, _ = stats.theilslopes(spectrum_subset["mz_annotated"], spectrum_subset["x1"])
lines_df["mz_estimated"] = mz_intercept + lines_df["x1"] * mz_slope




rs = RANSACRegressor()
rs.fit(
    X = spectrum_subset["x1"].to_numpy().reshape(-1,1),
    y = spectrum_subset["mz_annotated"])
lines_df["mz_estimated"] = rs.predict(lines_df["x1"].to_numpy().reshape((-1,1)))

plt.scatter(spectrum_subset["x1"], spectrum_subset["mz_annotated"])
plt.plot(lines_df["x1"], lines_df["mz_estimated"])

# Re-match to closest peak

tol_mz_match = 1
axis_rel_cutoff = 0.8
axis_y1max_cutoff = 0.02 # 10 % above the y=0 line we don't start new peaks
annotated_low_intensity = 0.05


def match_line_best(x, xref, tol):
    delta = np.abs(xref - x) 
    matching = delta < tol
    if sum(matching) > 0:
        return np.argmax(-delta)
    return np.nan

im_data["lines_index"] = im_data["mz_annotated"].map(lambda x:
                                               match_line_best(x, 
                                               xref = lines_df["mz_estimated"],
                                               tol = tol_mz_match))
im_data["lines_match"] = im_data["lines_index"]
spectrum_data_full = pd.merge(im_data, lines_df, how="outer", left_on='lines_index', right_index=True)

# TODO: Check if there is an obvious "y axis" that needs to be removed
# TODO: Check if any of the unannotated peaks are clearly not peaks (starting higher up)
y1max = spectrum_data_full.loc[spectrum_data_full["mz_annotated"].notna(), "y1"].max()
y1min = spectrum_data_full.loc[spectrum_data_full["mz_annotated"].notna(), "y2"].min()
y1limit = y1max - (y1max - y1min) * axis_y1max_cutoff
spectrum_data_full = spectrum_data_full.loc[
                                            (spectrum_data_full["y1"] > y1limit) | 
                                            (spectrum_data_full["mz_annotated"].notna())
                                            ]

spectrum_data = spectrum_data_full[["mz_annotated", "mz_estimated", "intensity_abs"]].copy()
spectrum_data["intensity_rel"] = spectrum_data["intensity_abs"] / spectrum_data["intensity_abs"].max()
spectrum_data["intensity_rel"].fillna(annotated_low_intensity, inplace=True)
spectrum_data["mz_coalesce"] = spectrum_data["mz_annotated"].fillna(spectrum_data["mz_estimated"])
spectrum_data.sort_values("mz_coalesce", inplace=True)
