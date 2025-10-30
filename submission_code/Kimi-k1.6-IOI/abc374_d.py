import math

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx +=1
    S = int(input[idx])
    idx +=1
    T = int(input[idx])
    idx +=1
    
    segments = []
    for _ in range(N):
        A = int(input[idx])
        B = int(input[idx+1])
        C = int(input[idx+2])
        D = int(input[idx+3])
        idx +=4
        segments.append( ((A,B), (C,D)) )
    
    # Collect all unique points including (0,0)
    points = set()
    points.add( (0, 0) )
    for seg in segments:
        points.add( seg[0] )
        points.add( seg[1] )
    points = list(points)
    point_to_idx = { p:i for i,p in enumerate(points) }
    initial_pos_idx = point_to_idx[ (0, 0) ]
    
    # Preprocess segments
    preprocessed = []
    for seg in segments:
        p1, p2 = seg
        s0 = point_to_idx[p1]
        s1 = point_to_idx[p2]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = math.hypot(dx, dy)
        print_time = length / T
        preprocessed.append( (s0, s1, length, print_time) )
    
    # Initialize DP
    INF = float('inf')
    dp = [ [ INF ] * len(points) for _ in range(1 << N) ]
    dp[0][initial_pos_idx] = 0.0
    
    for mask in range(1 << N):
        for pos_idx in range(len(points)):
            if dp[mask][pos_idx] == INF:
                continue
            # Try all segments not in mask
            for i in range(N):
                if not (mask & (1 << i)):
                    s0, s1, length, print_time = preprocessed[i]
                    for dir in [0, 1]:
                        if dir ==0:
                            start = s0
                            end = s1
                        else:
                            start = s1
                            end = s0
                        # Compute distance from current pos to start
                        x1, y1 = points[pos_idx]
                        x2, y2 = points[start]
                        dx = x2 - x1
                        dy = y2 - y1
                        distance = math.hypot(dx, dy)
                        move_time = distance / S
                        total_time = dp[mask][pos_idx] + move_time + print_time
                        new_mask = mask | (1 << i)
                        if total_time < dp[new_mask][end]:
                            dp[new_mask][end] = total_time
    
    full_mask = (1 << N) -1
    ans = min( dp[full_mask] )
    print("{0:.20f}".format(ans))
    
if __name__ == '__main__':
    main()