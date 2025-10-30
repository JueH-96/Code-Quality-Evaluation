def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    S = data[2]
    T = data[3]
    
    # Collect all valid "stamp" intervals [p, p+M-1] where T matches S exactly over length M
    intervals = []
    for start in range(N - M + 1):
        match = True
        for i in range(M):
            if S[start + i] != T[i]:
                match = False
                break
        if match:
            intervals.append((start, start + M - 1))
    
    # We want to cover [0..N-1] with these intervals using a greedy approach
    # Sort intervals by their start
    intervals.sort(key=lambda x: x[0])
    
    covered_end = -1  # The farthest we have covered so far
    idx = 0          # Pointer to intervals
    n_intervals = len(intervals)
    current_max_end = -1  # The best coverage we can achieve for the next step
    
    i = 0  # We'll move from left to right covering all positions
    while i < N:
        # Find at least one interval that starts <= i and extends coverage
        updated = False
        while idx < n_intervals and intervals[idx][0] <= i:
            # This interval can help cover position i
            if intervals[idx][1] > current_max_end:
                current_max_end = intervals[idx][1]
                updated = True
            idx += 1
        
        # If we couldn't find any interval that starts on/before i, can't cover position i
        if current_max_end < i:
            print("No")
            return
        
        # We can extend coverage to current_max_end
        covered_end = current_max_end
        i = covered_end + 1  # Jump beyond the newly covered region
    
    # If we exit the loop successfully, we've covered [0..N-1]
    print("Yes")

# Do not forget to call main()
if __name__ == "__main__":
    main()