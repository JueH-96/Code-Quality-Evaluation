def main():
    import sys
    from collections import deque

    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    K = int(next(it))
    P = [int(next(it)) for _ in range(N)]
    
    # For each value in the permutation, record its index.
    # Note: our indices are 0-indexed internally. That will not affect differences.
    pos = [0] * N
    for i, v in enumerate(P):
        pos[v - 1] = i  # Since v is in 1..N

    # The key observation:
    # A subsequence of indices is "good" if and only if the selected elements form the set of K consecutive integers.
    # Since the permutation contains each integer exactly once, for any consecutive block [a, a+K-1],
    # the only choice is to select the indices of these values.
    # The difference i_K - i_1 (in 0-indexed terms as max(pos[a:a+K]) - min(pos[a:a+K]))
    # remains the same as in 1-indexed form.
    # Thus, we need to compute for each a in {1,2,...,N-K+1} the value:
    #   diff(a) = max(pos[a-1], pos[a], ..., pos[a+K-2]) - min(pos[a-1], pos[a], ..., pos[a+K-2])
    # and answer the minimum diff among all a.
    # Let A = pos array.
    A = pos  # A is of length N, where A[i] = position of integer (i+1) in P.
    
    # We need an efficient sliding window minimum and maximum over A for window size = K.
    n = N
    window = K
    minDeque = deque()  # indices into A, will store indices in increasing order of A value.
    maxDeque = deque()  # similarly, store indices in decreasing order of A value.
    
    best = float('inf')
    
    for i in range(n):
        # Remove elements that are out of the current window for both deques.
        if minDeque and minDeque[0] <= i - window:
            minDeque.popleft()
        if maxDeque and maxDeque[0] <= i - window:
            maxDeque.popleft()
            
        # For the minimum deque: remove elements from the right with A value >= A[i]
        while minDeque and A[minDeque[-1]] >= A[i]:
            minDeque.pop()
        minDeque.append(i)
        
        # For the maximum deque: remove elements from the right with A value <= A[i]
        while maxDeque and A[maxDeque[-1]] <= A[i]:
            maxDeque.pop()
        maxDeque.append(i)
        
        # When we have a full window, compute the difference.
        if i >= window - 1:
            current_diff = A[maxDeque[0]] - A[minDeque[0]]
            if current_diff < best:
                best = current_diff

    sys.stdout.write(str(best))

if __name__ == "__main__":
    main()