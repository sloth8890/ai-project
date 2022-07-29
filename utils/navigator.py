from collections import deque

import cv2
import numpy as np
'''
    감도 조절 필요

    1. 체류시간이 짧으면 이동 (위, 아래)
    2. 아래로 체류시간이 길면 (좌/우 활성화)
    3. 좌/우 활성화 되면 체류시간 짧게 해서 이동 가능

    TODO : 좌우 이동키 매핑 방법?

'''

def baseline_logic(yaw, pitch):
    '''
        (return)
            ret: up, down, left, right, or None
    '''
    THRES_DOWN = -2.5
    THRES_UP = 0.0

    THRES_RIGHT = 5
    THRES_LEFT = -5

    up_down = None; left_right = None
    if pitch > THRES_UP:
        up_down = 'up'
    elif pitch < THRES_DOWN:
        up_down = 'down'

    if yaw > THRES_RIGHT:
        left_right = 'right'
    elif yaw < THRES_LEFT:
        left_right = 'left'

    return up_down, left_right

class Navigator:
    def __init__(self, hit_logic=baseline_logic):        
        self.hit_logic = hit_logic

        self._eyes = None
        self._action = None

        #! deprecated
        self.horizontal_trigger = False
        
        self._build_hyper_param()
        self._build_navi_map_param()

    def _build_navi_map_param(self):
        self.navi_height = 200
        self.navi_width = 200
        self.navi_center = (self.navi_width//2, self.navi_height//2)
        self._navi_map = np.zeros((self.navi_height, self.navi_width, 3), dtype=np.uint8) # black image

    def _build_hyper_param(self):
        # TODO : parse from args
        self.QUEUE_LENGTH = 15 #! dependent on fps, currently using 30 fps frame inputs
        self.UP_DOWN_RATIO = 0.5
        self.LEFT_RIGHT_RATIO = 0.5

        self.LR_TRIGGER_RATIO = 0.8

        self.que = {
            'up' : deque([False] * self.QUEUE_LENGTH*2), #! x2 
            'down' : deque([False] * self.QUEUE_LENGTH),
            'left' : deque([False] * self.QUEUE_LENGTH),
            'right' : deque([False] * self.QUEUE_LENGTH),
        }

        # generates temporal terms between gaze input
        self.INPUT_REJECT = 4
        self.input_cnt = 0


    def _reset_queue(self, key):
        self.que[key].clear()
        self.que[key] = deque([False] * self.QUEUE_LENGTH)
        # TODO: remove half instead of clear?

    #self.UP_DOWN_RATIO = 0.6
    #self.LR_TRIGGER_RATIO = 0.8
    def _criteria(self, key, ratio):
        return sum(self.que[key])/len(self.que[key]) > ratio
    
    @property
    def action(self):
        return self._action

    @action.setter
    def action(self, v):
        self._action = v

    @property
    def eyes(self):
        return self._eyes

    @eyes.setter
    def eyes(self, eye_image):
        self._eyes = eye_image
        
    @property
    def navi_map(self):
        navi_map = self._navi_map.copy()

        if self._eyes is not None:
            h, w = self._eyes.shape
            eyes = np.expand_dims(self._eyes, axis=2)
            eyes = np.concatenate((eyes, eyes, eyes), axis=2)

            start_w = (self.navi_width - w) // 2
            navi_map[:h, start_w:start_w+w, :] = eyes
        
        margin = 50

        if self._action == 'up':
            end_pt = (self.navi_width//2,  margin)
            color = (0,255,255)
        elif self._action == 'down':
            end_pt = (self.navi_width//2,  self.navi_height-margin)
            color = (0,0,255)
        elif self._action == 'left':
            end_pt = (margin, self.navi_height//2)
            color = (255,0,255)
        elif self._action == 'right':
            end_pt = (self.navi_width-margin, self.navi_height//2)
            color = (255,0,0)
        elif self._action == 'trigger':
            color = (255,255,0)
            navi_map = cv2.circle(navi_map, self.navi_center, 8, color, -1)
            return navi_map

        if self._action is not None:
            navi_map = cv2.arrowedLine(navi_map, self.navi_center, end_pt, color, 8)

        return navi_map

    def __call__(self, eval_result):
        yaw = eval_result.eye_yaw
        pitch = eval_result.eye_pitch

        action = None # ret
        if yaw is None or pitch is None:
            self._action = action
            return action

        up_down, left_right = self.hit_logic(yaw, pitch)

        
        # directly control 
        if up_down is not None: # up & down has higher priority than left & right
            action = up_down
        else:
            action = left_right

        if self.input_cnt < self.INPUT_REJECT:
            self.input_cnt += 1
            action = None
        else: 
            self.input_cnt = 0

        self._action = action
        return action

        # trigger based (left & right)
        if self.horizontal_trigger: # left right
            for k in self.que:
                if k == left_right:
                    self.que[k].appendleft(True)
                else:
                    self.que[k].appendleft(False)

                self.que[k].pop()
                assert len(self.que[k]) == self.QUEUE_LENGTH

            for k in ['left', 'right']:
                if self._criteria(k, self.LEFT_RIGHT_RATIO): # left right click
                    action = k
                    self._reset_queue(k) # reset
                
        else: # up down 
            for k in self.que:
                if k == up_down:
                    self.que[k].appendleft(True)
                else:
                    self.que[k].appendleft(False)

                self.que[k].pop()
                assert len(self.que[k]) == self.QUEUE_LENGTH

            for k in ['up', 'down']:
                if self._criteria(k, self.LR_TRIGGER_RATIO): # left right trigger on
                    action = 'trigger'
                    self.horizontal_trigger = True
                    self._reset_queue(k) # reset
                elif self._criteria(k, self.UP_DOWN_RATIO): # up down scroll
                    action = k
                    #self._reset_queue(k) # reset

        self._action = action
        return action
        '''
        # action 별 동작 시도
        if self.horizontal_trigger: # left right
            return action
                
        else: # up down
            if action == 'up':
                return action
            elif action == 'down':
                return action
            else:
                return None
        '''