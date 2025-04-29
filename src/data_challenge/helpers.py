import numpy as np
import cv2 as cv
import sys
import os



if __name__ == "__main__":
    img = cv.imread("./images/img1cropped.png")
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(img_gray, 122, 255, cv.THRESH_BINARY)
    cv.imshow('greyed image', thresh)
    cv.waitKey(0)
    cv.imwrite('img_greyed.jpg', thresh)
    cv.destroyAllWindows()
    
    contours, hierarchy = cv.findContours(image=thresh, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)
    thresh_color = cv.cvtColor(thresh, cv.COLOR_GRAY2BGR)
    cv.drawContours(thresh_color, contours, -1, (0, 255, 0), 1)
    cv.imshow('img_contoured', thresh_color)
    cv.waitKey(0)
    cv.imwrite('img_countoured.jpg', thresh_color)
    cv.destroyAllWindows()
    print("barnacles detected:", len(contours))
    
    adaptiveThresh = cv.adaptiveThreshold(img_gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 33, 3)
    cv.imshow('adaptive threshold', adaptiveThresh)
    cv.waitKey(0)
    cv.imwrite('adaptive_threshold.jpg', adaptiveThresh)
    cv.destroyAllWindows()

    contours, hierarchy = cv.findContours(image=adaptiveThresh, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)
    aThresh_color = cv.cvtColor(adaptiveThresh, cv.COLOR_GRAY2BGR)
    cv.drawContours(aThresh_color, contours, -1, (0, 255, 0), 1)
    cv.imshow('img_AT_contoured', aThresh_color)
    cv.waitKey(0)
    cv.imwrite('img_AT_countoured.jpg', aThresh_color)
    cv.destroyAllWindows()
    print("barnacles detected:", len(contours))


    mask = cv.imread("./images/mask1.png")
    mask_gray = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    ret, threshMask = cv.threshold(mask_gray, 150, 255, cv.THRESH_BINARY)
    cv.imshow('mask_greyed', threshMask)
    cv.waitKey(0)
    cv.imwrite('mask_greyed.jpg', threshMask)
    cv.destroyAllWindows()

    contours, hierarchy = cv.findContours(image=threshMask, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)
    maskCopy = mask.copy()
    cv.drawContours(maskCopy, contours, -1, (0, 255, 0), 3)
    cv.imshow('mask_contoured', maskCopy)
    cv.waitKey(0)
    cv.imwrite('mask_contoured.jpg', maskCopy)
    cv.destroyAllWindows()
    print("barnacles detected:", len(contours)-1)


