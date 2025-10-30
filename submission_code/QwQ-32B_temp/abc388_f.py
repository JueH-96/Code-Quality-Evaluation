def main():
    import sys
    N, M, A, B = map(int, sys.stdin.readline().split())
    bad_intervals = []
    for _ in range(M):
        L, R = map(int, sys.stdin.readline().split())
        bad_intervals.append((L, R))
    
    # Construct allowed regions
    regions = []
    prev_end = 1  # Start before the first region
    for (L, R) in bad_intervals:
        start = prev_end
        end = L - 1
        if start <= end:
            regions.append((start, end))
        prev_end = R + 1
    # Add the final region after last bad interval
    start = prev_end
    end = N
    if start <= end:
        regions.append((start, end))
    
    current_min = 1
    current_max = 1
    possible = True
    
    for region in regions:
        start, end = region
        new_min = max(start, current_min + A)
        new_max_candidate = min(end, current_max + B)
        if new_min > end:
            possible = False
            break
        # Determine new_max
        if new_min + A <= end:
            new_max = end
        else:
            new_max = new_min + B
        current_min = new_min
        current_max = new_max
        if current_max >= N:
            break  # Can't get better than this
    
    if possible and current_max >= N:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()