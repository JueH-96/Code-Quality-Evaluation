import math

def solve():
    a, b, c, d = map(int, input().split())
    x_coords = set()
    x_coords.add(a)
    x_coords.add(c)
    for i in range(a + 1, c):
        x_coords.add(i)
    
    y_coords = set()
    y_coords.add(b)
    y_coords.add(d)
    for i in range(b + 1, d):
        if i % 2 == 0:
            y_coords.add(i)
            
    sorted_x_coords = sorted(list(x_coords))
    sorted_y_coords = sorted(list(y_coords))
    
    unique_x_coords = []
    if sorted_x_coords:
        unique_x_coords.append(sorted_x_coords[0])
        for i in range(1, len(sorted_x_coords)):
            if sorted_x_coords[i] > sorted_x_coords[i-1]:
                unique_x_coords.append(sorted_x_coords[i])
    sorted_x_coords = unique_x_coords
    
    unique_y_coords = []
    if sorted_y_coords:
        unique_y_coords.append(sorted_y_coords[0])
        for i in range(1, len(sorted_y_coords)):
            if sorted_y_coords[i] > sorted_y_coords[i-1]:
                unique_y_coords.append(sorted_y_coords[i])
    sorted_y_coords = unique_y_coords
    
    black_area = 0
    
    for i in range(len(sorted_x_coords) - 1):
        x_start = sorted_x_coords[i]
        x_end = sorted_x_coords[i+1]
        for j in range(len(sorted_y_coords) - 1):
            y_start = sorted_y_coords[j]
            y_end = sorted_y_coords[j+1]
            
            mid_x = (x_start + x_end) / 2
            mid_y = (y_start + y_end) / 2
            
            index_val = math.floor(mid_x) + math.floor(mid_y / 2) + math.floor((mid_x + mid_y) / 2)
            
            if index_val % 2 == 0:
                black_area += (x_end - x_start) * (y_end - y_start)
                
    print(int(black_area * 2))

if __name__ == '__main__':
    solve()