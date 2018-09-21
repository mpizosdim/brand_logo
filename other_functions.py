from PIL import Image
from pytesseract import image_to_string
import os
import re
import psutil
from urllib.request import urlretrieve
from urllib.error import HTTPError
import time


def __isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


def __data_preprocess(path):
    brandnames = open(path, 'r').read().lower()
    data = re.sub(r'[^\w\s]', ' ', brandnames)
    examples = data.split("\n")
    examples = [re.sub(' +', ' ', x).strip() for x in examples]
    examples = [x for x in examples if __isEnglish(x)]
    brands = [x for x in examples if len(x) >= 2]
    return brands


def match_brands(path_of_logos, path_to_store, name_path):
    changed = 0
    count = 0
    brands_list = __data_preprocess(name_path)
    for file in os.listdir(path_of_logos):
        count = count + 1
        filename = path_of_logos + '%s' %file
        img = Image.open(filename)
        text = image_to_string(img)
        if text:
            text = text.lower()
            if text in brands_list:
                changed = changed + 1
                print("changed: %s. Processed: %s." %(changed, count))
                new_filename = path_to_store + '%s  .png' %text
                os.rename(filename, new_filename)


def name_an_image(path_of_logs, path_to_store):
    regexp = re.compile(r'\d.\d.png')
    for file in os.listdir(path_of_logs):
        if regexp.search(file):
            filename = path_of_logs + '%s' %file
            img = Image.open(filename)
            img.show()
            name = input("brand name?")
            if name == "x":
                img.close()
                os.remove(filename)
            elif name == "exit":
                break
            else:
                new_filename = path_to_store + '%s   .png' % name
                os.rename(filename, new_filename)
            for proc in psutil.process_iter():
                if proc.name() == "display":
                    proc.kill()


def logo_scrapper(start, end, url):
    start_time = time.time()
    count = 0
    for i in range(start, end):
        for j in [1, 2]:
            i_str = str(i)
            j_str = str(j)
            url_in = url+i_str+"."+j_str+".png"
            try:
                urlretrieve(url_in, "data/"+i_str+"."+j_str+".png")
                count = count + 1
                if count%50 == 0:
                    print("found %s pictures until now." %count)
            except HTTPError:
                a = 1+1
            if i%1000 == 0 and j == 1:
                elapsed_time = time.time() - start_time
                print("loop till: %s. Elapsed time: %s. Pictures found: %s" %(i, elapsed_time, count))
                start_time = time.time()


def main():
    name_an_image('./data', 'labeled_data/')
    #match_brands('./data', 'labeled_data/', 'brandnames.csv)
    #logo_scrapper(0, 100000, 'https://static.stylight.net/brands/de/res/162/')

if __name__ == '__main__':
    main()
