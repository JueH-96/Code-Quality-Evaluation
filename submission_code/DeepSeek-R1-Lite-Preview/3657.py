from typing import List

class Solution:
    def checkValidCuts(self, n: int, rectangles: List[List[int]]) -> bool:
        # Function to check for horizontal cuts
        def can_make_horizontal_cuts(n, rectangles):
            events = []
            for rect in rectangles:
                start_y = rect[1]
                end_y = rect[3]
                events.append((start_y, 'start'))
                events.append((end_y, 'end'))
            
            # Sort events: y coordinate, 'end' before 'start' at same y
            events.sort(key=lambda x: (x[0], 0 if x[1]=='end' else 1))
            
            active = 0
            cuts = []
            prev_y = 0
            for y, typ in events:
                if prev_y < y:
                    if active == 0:
                        cuts.append((prev_y, y))
                if typ == 'start':
                    active += 1
                else:
                    active -= 1
                prev_y = y
            if prev_y < n:
                if active == 0:
                    cuts.append((prev_y, n))
            
            # Now, choose two cuts and check if each section has at least one rectangle
            if len(cuts) < 2:
                return False
            
            # Precompute prefix counts
            rectangles_sorted = sorted(rectangles, key=lambda x: x[3])
            prefix = [0] * (len(rectangles_sorted) + 1)
            for i in range(len(rectangles_sorted)):
                prefix[i+1] = prefix[i] + 1
            
            # Check all pairs of cuts
            for i in range(len(cuts)):
                for j in range(i+1, len(cuts)):
                    a_start, a_end = cuts[i]
                    b_start, b_end = cuts[j]
                    # Section 1: y <= a_end
                    count1 = 0
                    left = 0
                    right = len(rectangles_sorted)
                    while left < right:
                        mid = (left + right) // 2
                        if rectangles_sorted[mid][3] > a_end:
                            right = mid
                        else:
                            left = mid + 1
                    count1 = left
                    
                    if count1 == 0 or count1 == len(rectangles_sorted):
                        continue
                    
                    # Section 3: y >= b_start
                    count3 = len(rectangles_sorted) - self.binary_search_upper(rectangles_sorted, b_start)
                    if count3 == 0 or count3 == len(rectangles_sorted):
                        continue
                    
                    # Section 2: a_end < y < b_start
                    count2 = len(rectangles_sorted) - count1 - count3
                    if count2 >= 1:
                        return True
            return False
        
        # Function to check for vertical cuts
        def can_make_vertical_cuts(n, rectangles):
            events = []
            for rect in rectangles:
                start_x = rect[0]
                end_x = rect[2]
                events.append((start_x, 'start'))
                events.append((end_x, 'end'))
            
            # Sort events: x coordinate, 'end' before 'start' at same x
            events.sort(key=lambda x: (x[0], 0 if x[1]=='end' else 1))
            
            active = 0
            cuts = []
            prev_x = 0
            for x, typ in events:
                if prev_x < x:
                    if active == 0:
                        cuts.append((prev_x, x))
                if typ == 'start':
                    active += 1
                else:
                    active -= 1
                prev_x = x
            if prev_x < n:
                if active == 0:
                    cuts.append((prev_x, n))
            
            # Now, choose two cuts and check if each section has at least one rectangle
            if len(cuts) < 2:
                return False
            
            # Precompute prefix counts
            rectangles_sorted = sorted(rectangles, key=lambda x: x[2])
            prefix = [0] * (len(rectangles_sorted) + 1)
            for i in range(len(rectangles_sorted)):
                prefix[i+1] = prefix[i] + 1
            
            # Check all pairs of cuts
            for i in range(len(cuts)):
                for j in range(i+1, len(cuts)):
                    a_start, a_end = cuts[i]
                    b_start, b_end = cuts[j]
                    # Section 1: x <= a_end
                    count1 = 0
                    left = 0
                    right = len(rectangles_sorted)
                    while left < right:
                        mid = (left + right) // 2
                        if rectangles_sorted[mid][2] > a_end:
                            right = mid
                        else:
                            left = mid + 1
                    count1 = left
                    
                    if count1 == 0 or count1 == len(rectangles_sorted):
                        continue
                    
                    # Section 3: x >= b_start
                    count3 = len(rectangles_sorted) - self.binary_search_upper(rectangles_sorted, b_start)
                    if count3 == 0 or count3 == len(rectangles_sorted):
                        continue
                    
                    # Section 2: a_end < x < b_start
                    count2 = len(rectangles_sorted) - count1 - count3
                    if count2 >= 1:
                        return True
            return False
        
        # Helper function for binary search
        def binary_search_upper(arr, value):
            left = 0
            right = len(arr)
            while left < right:
                mid = (left + right) // 2
                if arr[mid][0] > value:
                    right = mid
                else:
                    left = mid + 1
            return left
        
        # Check horizontal and vertical cuts
        return can_make_horizontal_cuts(n, rectangles) or can_make_vertical_cuts(n, rectangles)