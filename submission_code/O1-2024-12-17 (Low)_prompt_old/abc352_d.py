def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    N, K = map(int, input_data[:2])
    P = list(map(int, input_data[2:]))

    # pos[x] = index of x in permutation P (1-based)
    # We'll build pos so that pos[value] = position in P
    pos = [0] * (N + 1)
    for i in range(N):
        pos[P[i]] = i + 1  # store 1-based indices

    # We want to look at A[i] = pos[i] for i in 1..N
    # Then, for each window of length K in A (i..i+K-1), find
    # min and max. The difference of those is a candidate answer.
    # We'll use a deque-based sliding window min/max in O(N) time.

    A = [pos[i] for i in range(1, N + 1)]

    # Special case: If K == 1, answer is always 0
    if K == 1:
        print(0)
        return

    from collections import deque

    # Deques for min and max
    min_deq = deque()
    max_deq = deque()

    # Initialize the deques for the first K elements
    for i in range(K):
        # For min_deq
        while min_deq and A[min_deq[-1]] >= A[i]:
            min_deq.pop()
        min_deq.append(i)

        # For max_deq
        while max_deq and A[max_deq[-1]] <= A[i]:
            max_deq.pop()
        max_deq.append(i)

    min_diff = A[max_deq[0]] - A[min_deq[0]]  # difference for the first window

    # Slide over the array
    for i in range(K, N):
        # Pop from front if out of window
        while min_deq and min_deq[0] <= i - K:
            min_deq.popleft()
        while max_deq and max_deq[0] <= i - K:
            max_deq.popleft()

        # Push new index (i)
        while min_deq and A[min_deq[-1]] >= A[i]:
            min_deq.pop()
        min_deq.append(i)

        while max_deq and A[max_deq[-1]] <= A[i]:
            max_deq.pop()
        max_deq.append(i)

        # Current window is i-K+1..i
        current_diff = A[max_deq[0]] - A[min_deq[0]]
        if current_diff < min_diff:
            min_diff = current_diff

    print(min_diff)

def main():
    solve()

# Uncomment the following line to run solve() directly if needed
# solve()