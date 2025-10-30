import itertools
import math

def dist(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def solve():
    n, s, t = map(int, input().split())
    segments = []
    for _ in range(n):
        segments.append(list(map(int, input().split())))

    points = []
    for i in range(n):
        points.append((segments[i][0], segments[i][1]))
        points.append((segments[i][2], segments[i][3]))

    
    def calculate_time(order):
        total_time = 0
        current_x, current_y = 0, 0
        
        for i in range(n):
            seg_index = order[i] // 2
            start_point_index = order[i] % 2
            
            if start_point_index == 0:
                start_x, start_y = segments[seg_index][0], segments[seg_index][1]
                end_x, end_y = segments[seg_index][2], segments[seg_index][3]
            else:
                start_x, start_y = segments[seg_index][2], segments[seg_index][3]
                end_x, end_y = segments[seg_index][0], segments[seg_index][1]

            total_time += dist(current_x, current_y, start_x, start_y) / s
            total_time += dist(start_x, start_y, end_x, end_y) / t
            current_x, current_y = end_x, end_y
        return total_time

    min_time = float('inf')
    for order in itertools.permutations(range(2 * n)):
        
        valid = True
        for i in range(n):
            first_occurrence = -1
            second_occurrence = -1
            for j in range(2*n):
                if order[j] // 2 == i:
                    if first_occurrence == -1:
                        first_occurrence = j
                    else:
                        second_occurrence = j
                        break
            if first_occurrence == -1 or second_occurrence == -1:
                valid = False
                break
            if first_occurrence > second_occurrence:
                valid = False
                break
        
        if valid:
            min_time = min(min_time, calculate_time(order))

    print(min_time)

solve()