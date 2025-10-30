import sys, math
sys.setrecursionlimit(10**6)

def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    itr = iter(input_data)
    N = int(next(itr))
    S = float(next(itr))
    T = float(next(itr))
    
    # Each segment is represented as ((Ax, By), (Cx, Dy), length)
    segments = []
    for i in range(N):
        A = int(next(itr))
        B = int(next(itr))
        C = int(next(itr))
        D = int(next(itr))
        length = math.hypot(C - A, D - B)
        segments.append(((A, B), (C, D), length))
    
    # Use memoization: state = (mask, current_x, current_y)
    # mask indicates which segments are already printed.
    from functools import lru_cache
    @lru_cache(maxsize=None)
    def dp(mask, cx, cy):
        if mask == (1 << N) - 1:
            return 0.0
        best = float('inf')
        # Try printing each segment that is not yet printed
        for i in range(N):
            if not (mask & (1 << i)):
                (p1, p2, seg_len) = segments[i]
                # Option 1: Move to p1 (start point) then print to p2.
                travel_cost = math.hypot(cx - p1[0], cy - p1[1]) / S
                print_cost = seg_len / T
                total1 = travel_cost + print_cost + dp(mask | (1 << i), p2[0], p2[1])
                best = min(best, total1)
                
                # Option 2: Alternatively, move to p2 (start point) then print to p1.
                travel_cost = math.hypot(cx - p2[0], cy - p2[1]) / S
                total2 = travel_cost + print_cost + dp(mask | (1 << i), p1[0], p1[1])
                best = min(best, total2)
        return best

    ans = dp(0, 0, 0)
    sys.stdout.write(f"{ans:.12f}")

if __name__ == '__main__':
    main()