def main():
    import sys
    from collections import deque

    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    K = int(next(it))
    P = [int(next(it)) for _ in range(N)]
    
    # Create an array pos where, for each value x from 1 to N,
    # pos[x-1] is the index (0-indexed) at which x appears in P.
    pos = [0] * N
    for i, x in enumerate(P):
        pos[x - 1] = i

    # If K is 1, any one element forms a valid sequence with span 0.
    if K == 1:
        sys.stdout.write("0")
        return

    # The key observation is:
    # A good index sequence corresponds exactly to picking a block of consecutive 
    # integers {a, a+1, ..., a+K-1} and then taking their indices in P. 
    # The valid sequence (after sorting indices) will have span = max(pos[a...a+K-1]) - min(pos[a...a+K-1]).
    #
    # So, we need to compute for each consecutive block in the pos array of length K,
    # the difference between the maximum and minimum value.
    #
    # We can perform this efficiently using two sliding window deques: one for the window minimum
    # and one for the window maximum.

    dq_min = deque()
    dq_max = deque()
    ans = 10**9  # Initialize with a large number.

    # We are iterating over the pos array (which has length N) and for each window of K consecutive values in pos,
    # we will compute the span.
    for i in range(N):
        # Remove indices that have fallen out of the current window (which consists of indices [i-K+1, i]).
        while dq_min and dq_min[0] < i - K + 1:
            dq_min.popleft()
        while dq_max and dq_max[0] < i - K + 1:
            dq_max.popleft()

        # Maintain dq_min so that pos[dq_min] is in increasing order so that the front always gives the minimum.
        while dq_min and pos[dq_min[-1]] >= pos[i]:
            dq_min.pop()
        dq_min.append(i)

        # Similarly, maintain dq_max so that pos[dq_max] is in decreasing order so that the front always gives the maximum.
        while dq_max and pos[dq_max[-1]] <= pos[i]:
            dq_max.pop()
        dq_max.append(i)

        # Once we have formed a complete window (i.e. i >= K-1) compute the candidate span.
        if i >= K - 1:
            current_min = pos[dq_min[0]]
            current_max = pos[dq_max[0]]
            span = current_max - current_min
            if span < ans:
                ans = span

    sys.stdout.write(str(ans))

if __name__ == '__main__':
    main()