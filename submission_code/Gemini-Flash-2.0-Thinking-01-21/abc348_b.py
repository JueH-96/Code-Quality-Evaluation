import math

def solve():
    n = int(input())
    points = []
    for i in range(n):
        x, y = map(int, input().split())
        points.append({'x': x, 'y': y, 'id': i + 1})
    
    results = []
    for i in range(n):
        current_point = points[i]
        max_distance_squared = -1
        farthest_point_id = -1
        
        for j in range(n):
            if i == j:
                continue
            other_point = points[j]
            distance_squared = (current_point['x'] - other_point['x'])**2 + (current_point['y'] - other_point['y'])**2
            
            if distance_squared > max_distance_squared:
                max_distance_squared = distance_squared
                farthest_point_id = other_point['id']
            elif distance_squared == max_distance_squared:
                if farthest_point_id == -1 or other_point['id'] < farthest_point_id:
                    farthest_point_id = other_point['id']
                    
        results.append(farthest_point_id)
        
    for result_id in results:
        print(result_id)

if __name__ == '__main__':
    solve()