import math

def solve():
    n = int(input())
    points = []
    for _ in range(n):
        x, y = map(int, input().split())
        points.append((x, y))
    
    total_cost = 0.0
    current_x, current_y = 0, 0
    
    for i in range(n):
        next_x, next_y = points[i]
        distance = math.sqrt((current_x - next_x)**2 + (current_y - next_y)**2)
        total_cost += distance
        current_x, current_y = next_x, next_y
        
    distance_to_origin = math.sqrt((current_x - 0)**2 + (current_y - 0)**2)
    total_cost += distance_to_origin
    
    print(total_cost)

if __name__ == '__main__':
    solve()