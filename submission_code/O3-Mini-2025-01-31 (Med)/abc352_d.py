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
    
    # For each value x (1-indexed), store its position (also 1-indexed).
    pos = [0] * (N + 1)
    for i, x in enumerate(P):
        pos[x] = i + 1  # positions are 1-indexed

    # Notice that a good index sequence corresponds exactly to the set of indices
    # of a block of consecutive numbers [a, a+K-1] in the permutation.
    # Because the permutation contains each number exactly once,
    # for a fixed a (with 1 <= a <= N-K+1), the K numbers a, a+1, ..., a+K-1 appear in 
    # the permutation at positions pos[a], pos[a+1], ..., pos[a+K-1]. Their minimum and maximum 
    # positions give a candidate interval whose length (difference) is exactly what we get if we 
    # choose those indices. (Any subsequence that is a rearrangement of these consecutive numbers
    # must include these positions because they are the only appearances.)
    #
    # Thus, our job reduces to: For each a from 1 to N-K+1, compute:
    #   diff = max(pos[a...a+K-1]) - min(pos[a...a+K-1])
    # and output the minimum diff.
    #
    # Directly scanning over each block would be O(K) per block, which is too slow.
    # We use a sliding window technique to compute minimums and maximums over the array:
    # A = [pos[1], pos[2], ..., pos[N]]
    # for every window of length K.
    
    A = pos[1:]  # Now A[0] is pos[1], A[1] is pos[2], ..., length = N

    # Special case: if K == 1, every good sequence consists of one index and diff = 0
    if K == 1:
        sys.stdout.write("0")
        return

    n = N  # length of array A
    
    # Compute sliding window minimums for A with window size K.
    min_deque = deque()
    min_values = [0] * (n - K + 1)
    for i in range(n):
        # Remove from the right if the new value is smaller.
        while min_deque and A[min_deque[-1]] >= A[i]:
            min_deque.pop()
        min_deque.append(i)
        # Remove indices that are out of this window.
        if min_deque[0] <= i - K:
            min_deque.popleft()
        if i >= K - 1:
            min_values[i - K + 1] = A[min_deque[0]]
    
    # Compute sliding window maximums for A with window size K.
    max_deque = deque()
    max_values = [0] * (n - K + 1)
    for i in range(n):
        while max_deque and A[max_deque[-1]] <= A[i]:
            max_deque.pop()
        max_deque.append(i)
        if max_deque[0] <= i - K:
            max_deque.popleft()
        if i >= K - 1:
            max_values[i - K + 1] = A[max_deque[0]]
    
    # Compute the answer as the minimal difference among all windows.
    ans = float('inf')
    for i in range(n - K + 1):
        diff = max_values[i] - min_values[i]
        if diff < ans:
            ans = diff
    sys.stdout.write(str(ans))

if __name__ == '__main__':
    main()