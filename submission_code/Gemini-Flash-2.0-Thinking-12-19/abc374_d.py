import math

def solve():
    n, s, t = map(int, input().split())
    segments = []
    points = [(0, 0)]
    segment_endpoints = []
    for _ in range(n):
        a, b, c, d = map(int, input().split())
        segments.append(((a, b), (c, d)))
        points.append((a, b))
        points.append((c, d))
        segment_endpoints.append(((a, b), (c, d)))
    
    unique_points = []
    point_to_index = {}
    index = 0
    for point in points:
        if point not in point_to_index:
            unique_points.append(point)
            point_to_index[point] = index
            index += 1
            
    num_unique_points = len(unique_points)
    distance_matrix = [[0.0] * num_unique_points for _ in range(num_unique_points)]
    for i in range(num_unique_points):
        for j in range(num_unique_points):
            p1 = unique_points[i]
            p2 = unique_points[j]
            distance = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            distance_matrix[i][j] = distance
            
    memo = {}
    
    def get_min_time(current_pos_index, segments_mask):
        if segments_mask == 0:
            return 0
        if (current_pos_index, segments_mask) in memo:
            return memo[(current_pos_index, segments_mask)]
        
        min_time = float('inf')
        for i in range(n):
            if (segments_mask >> i) & 1:
                segment = segment_endpoints[i]
                p1, p2 = segment[0], segment[1]
                p1_index = point_to_index[p1]
                p2_index = point_to_index[p2]
                
                # direction 1: p1 -> p2
                move_time_1 = distance_matrix[current_pos_index][p1_index] / s
                print_time_1 = distance_matrix[p1_index][p2_index] / t
                remaining_segments_mask_1 = segments_mask & ~(1 << i)
                future_time_1 = get_min_time(p2_index, remaining_segments_mask_1)
                total_time_1 = move_time_1 + print_time_1 + future_time_1
                min_time = min(min_time, total_time_1)
                
                # direction 2: p2 -> p1
                move_time_2 = distance_matrix[current_pos_index][p2_index] / s
                print_time_2 = distance_matrix[p2_index][p1_index] / t
                remaining_segments_mask_2 = segments_mask & ~(1 << i)
                future_time_2 = get_min_time(p1_index, remaining_segments_mask_2)
                total_time_2 = move_time_2 + print_time_2 + future_time_2
                min_time = min(min_time, total_time_2)
                
        memo[(current_pos_index, segments_mask)] = min_time
        return min_time
        
    initial_pos_index = point_to_index[(0, 0)]
    initial_segments_mask = (1 << n) - 1
    result = get_min_time(initial_pos_index, initial_segments_mask)
    print(f"{result:.12f}")

if __name__ == '__main__':
    solve()