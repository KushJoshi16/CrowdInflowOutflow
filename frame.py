
import numpy as np


class Frame:
    """
    This class is used to store the state of the frame
    State 0 means that the frame is not processed
    State 1 means that the frame is processed
    if the frame is not processed there wont be any tracks and only boxes will be present 
    and we will store the detections in place of tracks


    if any object goes undetected in current frame then 
    the last tracks_status of the object can give us the number of frames required for object to be considered as lost out of area
    once the object is lost we no more need to track it and count it
    but if the object is not lost it should be kept in counting if it comes back in tracking and is within area.

    """

    frame_state = None
    tracks = None
    tracks_state = None
    area_coordinates = None
    state_class = ["Out","In","Inflowing","Outflowing"]


    def __init__(self, area_coordinates, frame_state = 0, prev_frame = None):
        self.frame_state = frame_state
        self.tracks = []
        self.tracks_state = dict()
        self.area_coordinates = area_coordinates
        # self.area_coordinates = (x1_area, y1_area, x2_area, y2_area)
    
    def append(self, track):
        self.tracks.append(track)
    
    def update(self, tracks):
        self.tracks = tracks
        for track in tracks:
            state = self.calculate_state(track)
            self.update_state(track.track_id, state)
            
    def boxAndAreaOverlap(self,x_mid_point, y_mid_point, area_coordinates):
        x1_area, y1_area, x2_area, y2_area = area_coordinates #Unpacking

        if (x_mid_point >= x1_area and x_mid_point <= x2_area) and\
            (y_mid_point >= y1_area and y_mid_point <= y2_area):
            return True
        return False

    def InflowOutflowCheck(self, mean, area_coordinates):
        pos = mean[:2]
        vx,vy = mean[4:6] # velocity x and y components
        x1_area, y1_area, x2_area, y2_area = area_coordinates #Unpacking
        x_direction = vx/abs(vx)
        y_direction = vy/abs(vy)

        x_in_direction = (x2_area-pos[0]) - (pos[0]-x1_area)
        x_in_direction = x_in_direction/abs(x_in_direction)

        y_in_direction = (y2_area-pos[1]) - (pos[1]-y1_area)
        y_in_direction = y_in_direction/abs(y_in_direction)

        """
        for inflow check:
            check all the traffic that overlaps in 5 percent ouside the frame
        for outflow check:
            check all the traffic that overlaps in 5 percent inside the frame
        """

        inflow_chek_box = ((x1_area*92)//100, (y1_area*92)//100, (x2_area*108)//100, (y2_area*108)//100)

        outflow_chek_box = ((x1_area*108)//100, (y1_area*108)//100, (x2_area*92)//100, (y2_area*92)//100)

        if self.boxAndAreaOverlap(pos[0], pos[1], area_coordinates):
            if self.boxAndAreaOverlap(pos[0], pos[1], outflow_chek_box):
                # check direction of the object and return accordingly
                return 1
            else:
                if x_direction == x_in_direction and y_direction == y_in_direction:
                    return 2
                else:
                    return 3
        elif self.boxAndAreaOverlap(pos[0], pos[1], inflow_chek_box):
            if x_direction == x_in_direction and y_direction == y_in_direction:
                return 2
            else:
                return 0
        else:
            return 0



    def calculate_state(self, track):
        """
        returns : state tuple haing flowstate and n_frame
            (flow_state,n_frame)
        flow_state:
            0: object is out of area
            1: object is in area
            2: object is inflowing
            3: object is outflowing
        """
        flow_state = None
        
        x_mid, y_mid = track.mean[:2]
        # if self.boxAndAreaOverlap(x_mid, y_mid, self.area_coordinates):
        #     flow_state = 1

        # else:
        #     flow_state = 0
        flow_state = self.InflowOutflowCheck(track.mean, self.area_coordinates)
        return np.array([flow_state,1])

    def update_state(self, track_id, state):
        """
        state will contain:
        0: object is out of area
        1: object is in area
        2: object is inflowing
        3: object is outflowing

        n_frame : frames for which the object will remain inside area at current velocity
        will be calculated at each frame

        if frame state is zero the n_frame will not be calculated as there is no velocity of the object yet
        // [X] in such case box will be used to keep track of the object
        we are not tracking the objects that have not been tracked yet... Let them be in the frame uncounted
        """
        # if self.tracks_state is not None:
        self.tracks_state[track_id] = state

    # def get_count(self):
    #     return len(self.tracks)

