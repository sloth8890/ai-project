import time
import os

#import pyautogui

from selenium import webdriver
from selenium.webdriver.common.keys import Keys


class Action:
    scroll = {
        'down' : Keys.PAGE_DOWN,
        'up' : Keys.PAGE_UP,
        'left' : None,
        'right' : None,
        'back' : Keys.BACK_SPACE,
    }
    click = {
        'left': (1366, 722),
        'right': (1391, 727),
    }
    
class URLs:
    url = {
        'webtoon' : 'https://comic.naver.com/webtoon/detail?titleId=795487&no=12',
        'webtoon2' : 'https://comic.naver.com/webtoon/detail?titleId=796218&no=10',
    }


class WebController:
    def __init__(self, url:str=None, method:str='chromedriver', bin_root:str='/usr/local/bin'):
        self._build(url, method, bin_root)
        
    def _build(self, url:str, method:str, bin_root:str) -> None:
        self.url = URLs.url[url]

        if method == 'chromedriver':
            self.driver = webdriver.Chrome(os.path.join(bin_root, method))
            #self.driver.get(self.url)
            #self.driver.implicitly_wait(3)        
        else:
            raise NotImplementedError

    def open_url(self):
        self.driver.get(self.url)
        self.driver.implicitly_wait(3)        

    def __call__(self, key: str) -> None:
        if key is None:
            return 
        key = key.lower()
        if key in ['up', 'down']:
            if key == 'down':
                self.driver.find_element_by_tag_name('body').send_keys(Action.scroll[key])
            
            self.driver.find_element_by_tag_name('body').send_keys(Action.scroll[key])


        elif key in ['left', 'right']:
            if key == 'left':
                self.driver.find_element_by_xpath('//*[@id="comicRemocon"]/div[2]/div[1]/a[2]').click()
            elif key == 'right':
                self.driver.find_element_by_xpath('//*[@id="comicRemocon"]/div[2]/div[1]/a[3]').click()
            # mouse control 
            ''' # deprecated
            pyautogui.moveTo(*Action.click[key])
            time.sleep(0.5)
            pyautogui.click()
            time.sleep(0.5)
            '''

def main():
    web_controller = WebController(url='webtoon')
    web_controller.open_url()
    
    #time.sleep(5.0)    

    import cv2
    import numpy as np
    image = np.zeros((500, 500, 3))

    while True:
        cv2.imshow('remocon', image)
        key = cv2.waitKey(0)

        if key == ord('q'):
            break
        elif key == ord('w'):
            web_controller('up')
        elif key == ord('s'):
            web_controller('down')
        elif key == ord('a'):
            web_controller('left')
        elif key == ord('d'):
            web_controller('right')
            
    
if __name__ == '__main__':
    main()