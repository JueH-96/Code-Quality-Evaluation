# YOUR CODE HERE
import sys
import bisect
from collections import defaultdict

def main():
    import sys
    import sys
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    N = int(data[idx]); idx +=1
    M = int(data[idx]); idx +=1
    Sx = int(data[idx]); idx +=1
    Sy = int(data[idx]); idx +=1
    
    houses_by_x = defaultdict(list)
    houses_by_y = defaultdict(list)
    
    for _ in range(N):
        X = int(data[idx]); idx +=1
        Y = int(data[idx]); idx +=1
        houses_by_x[X].append(Y)
        houses_by_y[Y].append(X)
    
    # Sort the lists for binary search
    for x in houses_by_x:
        houses_by_x[x].sort()
    for y in houses_by_y:
        houses_by_y[y].sort()
    
    passed_houses = set()
    
    current_x, current_y = Sx, Sy
    
    for _ in range(M):
        D = data[idx]; idx +=1
        C = int(data[idx]); idx +=1
        if D == 'U':
            new_x = current_x
            new_y = current_y + C
            # Vertical movement
            x0 = current_x
            y_start = current_y
            y_end = new_y
            lower = min(y_start, y_end)
            upper = max(y_start, y_end)
            if x0 in houses_by_x:
                y_list = houses_by_x[x0]
                left = bisect.bisect_left(y_list, lower)
                right = bisect.bisect_right(y_list, upper)
                for y in y_list[left:right]:
                    passed_houses.add( (x0, y) )
            current_y = new_y
        elif D == 'D':
            new_x = current_x
            new_y = current_y - C
            # Vertical movement
            x0 = current_x
            y_start = current_y
            y_end = new_y
            lower = min(y_start, y_end)
            upper = max(y_start, y_end)
            if x0 in houses_by_x:
                y_list = houses_by_x[x0]
                left = bisect.bisect_left(y_list, lower)
                right = bisect.bisect_right(y_list, upper)
                for y in y_list[left:right]:
                    passed_houses.add( (x0, y) )
            current_y = new_y
        elif D == 'R':
            new_x = current_x + C
            new_y = current_y
            # Horizontal movement
            y0 = current_y
            x_start = current_x
            x_end = new_x
            lower = min(x_start, x_end)
            upper = max(x_start, x_end)
            if y0 in houses_by_y:
                x_list = houses_by_y[y0]
                left = bisect.bisect_left(x_list, lower)
                right = bisect.bisect_right(x_list, upper)
                for x in x_list[left:right]:
                    passed_houses.add( (x, y0) )
            current_x = new_x
        elif D == 'L':
            new_x = current_x - C
            new_y = current_y
            # Horizontal movement
            y0 = current_y
            x_start = current_x
            x_end = new_x
            lower = min(x_start, x_end)
            upper = max(x_start, x_end)
            if y0 in houses_by_y:
                x_list = houses_by_y[y0]
                left = bisect.bisect_left(x_list, lower)
                right = bisect.bisect_right(x_list, upper)
                for x in x_list[left:right]:
                    passed_houses.add( (x, y0) )
            current_x = new_x
        else:
            # Invalid direction, skip
            pass
    
    print(current_x, current_y, len(passed_houses))

if __name__ == "__main__":
    main()