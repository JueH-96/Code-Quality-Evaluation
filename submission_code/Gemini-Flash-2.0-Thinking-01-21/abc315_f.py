import math

def solve():
    n = int(input())
    points = []
    for _ in range(n):
        x, y = map(int, input().split())
        points.append((x, y))
    
    if n <= 1:
        print(0.0)
        return
        
    distances = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dist = math.sqrt((points[i][0] - points[j][0])**2 + (points[i][1] - points[j][1])**2)
            distances[i][j] = dist
            distances[j][i] = dist
            
    dp = {} # dp[(checkpoint_index, jumps_count)] = min_distance
    dp[(0, 0)] = 0.0
    
    min_total_cost = float('inf')
    
    for jumps in range(1, n):
        for current_checkpoint_index in range(1, n):
            min_dist_to_current = float('inf')
            for prev_checkpoint_index in range(current_checkpoint_index):
                if (prev_checkpoint_index, jumps - 1) in dp:
                    dist = distances[prev_checkpoint_index][current_checkpoint_index]
                    prev_min_dist = dp[(prev_checkpoint_index, jumps - 1)]
                    current_dist = prev_min_dist + dist
                    min_dist_to_current = min(min_dist_to_current, current_dist)
            if min_dist_to_current != float('inf'):
                dp[(current_checkpoint_index, jumps)] = min_dist_to_current
                
    for jumps in range(1, n):
        if (n - 1, jumps) in dp:
            distance_traveled = dp[(n - 1, jumps)]
            skipped_checkpoints_count = n - (jumps + 1)
            penalty = 0
            if skipped_checkpoints_count > 0:
                penalty = 2**(skipped_checkpoints_count - 1)
            total_cost = distance_traveled + penalty
            min_total_cost = min(min_total_cost, total_cost)
            
    print(f"{min_total_cost:.12f}")

if __name__ == '__main__':
    solve()