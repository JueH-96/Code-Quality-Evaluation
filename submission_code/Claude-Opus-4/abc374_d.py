import math
from itertools import permutations

def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def solve():
    # Read input
    N, S, T = map(int, input().split())
    segments = []
    for _ in range(N):
        a, b, c, d = map(int, input().split())
        segments.append(((a, b), (c, d)))
    
    # For each segment, calculate the time to print it
    print_times = []
    for (x1, y1), (x2, y2) in segments:
        print_times.append(distance(x1, y1, x2, y2) / T)
    
    # Dynamic programming with bitmask
    # dp[mask][i][endpoint] = minimum time to print segments in mask, ending at segment i's endpoint
    # endpoint: 0 = first endpoint, 1 = second endpoint
    INF = float('inf')
    dp = {}
    
    # Initialize: start from (0, 0) and print first segment
    for i in range(N):
        mask = 1 << i
        # Starting from endpoint 0 of segment i
        time0 = distance(0, 0, segments[i][0][0], segments[i][0][1]) / S + print_times[i]
        dp[(mask, i, 1)] = time0  # End at endpoint 1
        
        # Starting from endpoint 1 of segment i
        time1 = distance(0, 0, segments[i][1][0], segments[i][1][1]) / S + print_times[i]
        dp[(mask, i, 0)] = time1  # End at endpoint 0
    
    # Fill DP table
    for mask in range(1, 1 << N):
        for last in range(N):
            if not (mask & (1 << last)):
                continue
            
            for last_end in range(2):
                if (mask, last, last_end) not in dp:
                    continue
                
                current_time = dp[(mask, last, last_end)]
                last_pos = segments[last][last_end]
                
                # Try adding each unvisited segment
                for next_seg in range(N):
                    if mask & (1 << next_seg):
                        continue
                    
                    new_mask = mask | (1 << next_seg)
                    
                    # Try both orientations of the next segment
                    # Start from endpoint 0
                    move_time0 = distance(last_pos[0], last_pos[1], 
                                         segments[next_seg][0][0], segments[next_seg][0][1]) / S
                    new_time0 = current_time + move_time0 + print_times[next_seg]
                    key0 = (new_mask, next_seg, 1)
                    if key0 not in dp or dp[key0] > new_time0:
                        dp[key0] = new_time0
                    
                    # Start from endpoint 1
                    move_time1 = distance(last_pos[0], last_pos[1], 
                                         segments[next_seg][1][0], segments[next_seg][1][1]) / S
                    new_time1 = current_time + move_time1 + print_times[next_seg]
                    key1 = (new_mask, next_seg, 0)
                    if key1 not in dp or dp[key1] > new_time1:
                        dp[key1] = new_time1
    
    # Find minimum time among all possible endings
    full_mask = (1 << N) - 1
    min_time = INF
    for i in range(N):
        for end in range(2):
            if (full_mask, i, end) in dp:
                min_time = min(min_time, dp[(full_mask, i, end)])
    
    print(f"{min_time:.20f}")

solve()