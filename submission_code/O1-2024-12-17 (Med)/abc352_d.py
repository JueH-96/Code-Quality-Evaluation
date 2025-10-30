def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N, K = map(int, input_data[:2])
    P = list(map(int, input_data[2:]))

    # pos[x-1] will store the position of the integer x in the permutation P (0-based)
    pos = [0]*N
    for i, val in enumerate(P):
        pos[val - 1] = i

    # We will perform a sliding window of size K over pos[0..N-1]
    # and compute the minimum of (max_in_window - min_in_window).
    from collections import deque
    
    # Deques for indices of elements in 'pos' to track min and max in O(1) per step
    minD = deque()  # will store indices in increasing order of pos' values
    maxD = deque()  # will store indices in decreasing order of pos' values
    
    min_diff = float('inf')
    
    for i in range(N):
        # Remove indices out of the current window (i-K+1 .. i)
        while minD and minD[0] <= i - K:
            minD.popleft()
        while maxD and maxD[0] <= i - K:
            maxD.popleft()
        
        # Maintain monotonic property for the min deque
        while minD and pos[minD[-1]] >= pos[i]:
            minD.pop()
        minD.append(i)
        
        # Maintain monotonic property for the max deque
        while maxD and pos[maxD[-1]] <= pos[i]:
            maxD.pop()
        maxD.append(i)
        
        # Once we've covered at least K elements, compute the difference
        if i + 1 >= K:
            curr_min = pos[minD[0]]
            curr_max = pos[maxD[0]]
            diff = curr_max - curr_min
            if diff < min_diff:
                min_diff = diff
    
    print(min_diff)

# Don't forget to call main()
if __name__ == "__main__":
    main()