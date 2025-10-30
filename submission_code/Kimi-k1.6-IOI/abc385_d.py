import bisect
import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr +=1
    M = int(input[ptr])
    ptr +=1
    Sx = int(input[ptr])
    ptr +=1
    Sy = int(input[ptr])
    ptr +=1
    
    # Read houses
    x_to_ys = dict()
    y_to_xs = dict()
    for _ in range(N):
        x = int(input[ptr])
        ptr +=1
        y = int(input[ptr])
        ptr +=1
        if x not in x_to_ys:
            x_to_ys[x] = []
        x_to_ys[x].append(y)
        if y not in y_to_xs:
            y_to_xs[y] = []
        y_to_xs[y].append(x)
    
    # Sort the lists
    x_keys = list(x_to_ys.keys())
    for x in x_keys:
        x_to_ys[x].sort()
    y_keys = list(y_to_xs.keys())
    for y in y_keys:
        y_to_xs[y].sort()
    
    current_x = Sx
    current_y = Sy
    visited = set()
    
    # Read movements
    for _ in range(M):
        direction = input[ptr]
        ptr +=1
        c = int(input[ptr])
        ptr +=1
        
        new_x = current_x
        new_y = current_y
        if direction == 'U':
            new_y += c
        elif direction == 'D':
            new_y -= c
        elif direction == 'L':
            new_x -= c
        elif direction == 'R':
            new_x += c
        
        # Process the segment
        if current_x == new_x:
            # Vertical movement
            y_start = current_y
            y_end = new_y
            min_y = min(y_start, y_end)
            max_y = max(y_start, y_end)
            x = current_x
            if x in x_to_ys:
                ys = x_to_ys[x]
                left = bisect.bisect_left(ys, min_y)
                right = bisect.bisect_right(ys, max_y)
                for y in ys[left:right]:
                    house = (x, y)
                    if house not in visited:
                        visited.add(house)
        else:
            # Horizontal movement
            x_start = current_x
            x_end = new_x
            min_x = min(x_start, x_end)
            max_x = max(x_start, x_end)
            y = current_y
            if y in y_to_xs:
                xs = y_to_xs[y]
                left = bisect.bisect_left(xs, min_x)
                right = bisect.bisect_right(xs, max_x)
                for x_coord in xs[left:right]:
                    house = (x_coord, y)
                    if house not in visited:
                        visited.add(house)
        
        # Update current position
        current_x, current_y = new_x, new_y
    
    print(current_x, current_y, len(visited))

if __name__ == "__main__":
    main()