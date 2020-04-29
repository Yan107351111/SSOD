# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 18:28:55 2020

@author: yan10
"""
import cv2
import os
import pickle
import sys
from tqdm import tqdm
import webvtt
from word_forms.word_forms import get_word_forms

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

def forms2list(dict_):
    list_ = []
    for key_ in list(dict_):
        list_ += list(dict_[key_])
    return list(set(list_))

def listIntersection(lst0, lst1):
    '''
    Find the intersection of list lst0 and lst1.
    -------
    params:
    -------
    lst0:  list of key strings.
    lst1:  string.
    -------
    return:
    -------
    list of values contained in both lists.
    '''
    values = []
    for value in lst0:
        sub_values = forms2list(get_word_forms(value))
        
        for sub_value in sub_values:
            if sub_value in lst1:
                values.append(value)
                break
    return values

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

def extractFrames(in_dir, out_dir, search_words, target_list = []
                  checkpoint = 'extract_checkpoint.p', null_class = 'negative'):
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
    if checkpoint in os.listdir():
        timeDict = pickel.load(checkpoint, "rb"))
    else:
        timeDict = dict()
        # List the subtitle files
        vttFiles = [i for i in os.listdir(path) if i[-3:] == 'vtt']
        print('Searching subtitles...')
        for vttFile in tqdm(vttFiles):
            # Go over each subtitle file and list the time stamps with at least
            # one of the searched words.
            found = False
            for caption in webvtt.read(os.path.join(path, vttFile)):
                string_list = stringSplit(caption.text, ' ', ',', '\n', '.')
                intrs = listIntersection(search_words, caption.text.lower())
                if intrs:
                    found = True
                    start = time2secs(caption.start)
                    end   = time2secs(caption.end)
                    time_ = (start+end)/2
                    if vttFile in list(timeDict):
                        processed = True
                        if vttFile in target_list or not target_list:
                            processed = False
                        timeDict[vttFile].append(processed)
                        timeDict[vttFile].append((time_, intrs))
                    else:
                        timeDict[vttFile] = [(time_, intrs),]
            if not found:
                for caption in webvtt.read(os.path.join(path, vttFile)):
                    start = time2secs(caption.start)
                    end   = time2secs(caption.end)
                    time_ = (start+end)/2
                    if vttFile in list(timeDict):
                        processed = True
                        if vttFile in target_list or not target_list:
                            processed = False
                        timeDict[vttFile].append(processed)
                        timeDict[vttFile].append((time_, [null_class,]))
                    else:
                        timeDict[vttFile] = [(time_, [null_class,]),]
        pickle.dump(timeDict, open(checkpoint, "wb"))
    
    print('Searching videos...')
    # extract the frames according to the found words and tag them.
    for file in tqdm(list(timeDict)):
        if timeDict[file][0]:
            continue
        timeDict[file][0] = True
        pickle.dump(timeDict, open(checkpoint, "wb"))
        time_list  = timeDict[file][1:]
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
                    fname = word+'_'+file[:-7]+f'_{round(time_)}'+'.png'
                    cv2.imwrite(os.path.join(out_dir, fname), frame)
                if time_list:
                    next_time, words = time_list.pop()
                else: break 
            ret, frame = video.read()

if __name__ == "__main__"    :
    # path = 'example'
    # search_words = ['book', 'Book','books', 'Books',
    #                 'pillow', 'Pillow', 'pillows', 'Pillows',
    #                 'leaf','Leaf', 'leaves', 'Leaves']
    # out_dir = 'frames'
    
    assert len(sys.argv)>=3, AssertionError('paths to source and desination required')
    path         = sys.argv[1]
    out_dir      = sys.argv[2]
    if len(sys.argv) < 4:
        search_words = ['bike', 'cup', 'dog', 'drum', 'guitar',
                        'gun', 'horse', 'pan', 'plate',
                        'scissors', 'tire']
    else:
        search_words = []
        for i in range(4, len(sys.argv)):
            search_words.append(sys.argv[i])
    target_list = []
    if sys.argv>=3:
        t = open(sys.argv[3], 'r')
        line = t.readline()
        while line:
            target_list.append(line[:-1])
            print(f'registered target {line}')
            line = t.readline()
        t.close()
    extractFrames(path, out_dir, search_words, target_list = target_list)
    
    
    
    
    
    
    
