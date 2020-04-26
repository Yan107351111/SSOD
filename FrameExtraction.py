# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 18:28:55 2020

@author: yan10
"""

import os
import webvtt
import cv2

def time2secs(string):
    '''
    conver the string string of format [hh:mm:ss.mls] to a float of seconds.
    -------
    params:
    -------
    string:  time in the format [hh:mm:ss.mls].
    -------
    return:
    -------
    time in the format of a float.
    '''
    return sum([60**(2-ii)*float(i) for ii, i in enumerate(string.split(':'))])

def listIntersection(lst0, lst1):
    '''
    Find the intersection of list lst0 and lst1.
    -------
    params:
    -------
    lst0:  list.
    lst1:  another list.
    -------
    return:
    -------
    list of values contained in both lists.
    '''
    return [value for value in lst0 if value in lst1]

def stringSplit(string, delimiter, *args):
    '''
    Split thestring string to a list of substrings by the delimiters delimiter
    and *args.
    -------
    params:
    -------
    string:     string to be split.
    delimiter:  delimiter to split by.
    *args:      aditional delimiters to split by.
    -------
    return:
    -------
    None
    '''
    for arg in args:
        string = string.replace(arg, delimiter)
    return string.split(' ')

def extractFrames(in_dir, out_dir, search_words, null_class = 'negative'):
    '''
    Get the frames coresponding to the subtitles containing one of the 
    searched words. 
    -------
    params:
    -------
    in_dir:         folder containing the subtitle files.
    out_dir:        destination for the image files.
    search_words:   words to search for in the subtitles.
    -------
    return:
    -------
    None
    '''
    timeDict = dict()
    # List the subtitle files
    vttFiles = [i for i in os.listdir(path) if i[-3:] == 'vtt']
    for vttFile in vttFiles:
        # Go over each subtitle file and list the time stamps with at least
        # one of the searched words.
        found = False
        for caption in webvtt.read(os.path.join(path, vttFile)):
            string_list = stringSplit(caption.text, ' ', ',', '\n', '.')
            intrs = listIntersection(string_list, search_words)
            if intrs:
                found = True
                start = time2secs(caption.start)
                end   = time2secs(caption.end)
                time_ = (start+end)/2
                if vttFile in list(timeDict):
                    timeDict[vttFile].append((time_, intrs))
                else:
                    timeDict[vttFile] = [(time_, intrs),]
        if not found:
            for caption in webvtt.read(os.path.join(path, vttFile)):
                start = time2secs(caption.start)
                end   = time2secs(caption.end)
                time_ = (start+end)/2
                if vttFile in list(timeDict):
                    timeDict[vttFile].append((time_, [null_class,]))
                else:
                    timeDict[vttFile] = [(time_, [null_class,]),]

    # extract the frames according to the found words and tag them.
    for file in list(timeDict):
        time_list  = timeDict[file]
        time_list.reverse()
        video_file = file[:-7]+'.mp4'
        video      = cv2.VideoCapture(os.path.join(path, video_file))
        fps        = video.get(cv2.CAP_PROP_FPS)
        ret, frame = video.read()
        frameNum   = 1
        next_time, words = time_list.pop()
        while ret:
            time_ = frameNum*1/fps
            frameNum+=1
            if time_>=next_time:
                for word in words:
                    fname = word+file[:-7]+f'_{round(time_)}'+'.png'
                    cv2.imwrite(os.path.join(out_dir, fname), frame)
                if time_list:
                    next_time, words = time_list.pop()
                else: break 
            ret, frame = video.read()

if __name__ == "__main__"    :
    path = 'example'
    search_words = ['book', 'Book','books', 'Books',
                    'pillow', 'Pillow', 'pillows', 'Pillows',
                    'leaf','Leaf', 'leaves', 'Leaves']
    out_dir = 'frames'
    
    extractFrames(path, out_dir, search_words)
    
    
    
    
    
    
    
    
