import sys
import threading

def main():
    import sys
    from collections import deque

    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    K = int(next(it))
    P = [int(next(it)) for _ in range(N)]

    # pos[x] = index of value x in P (0-based)
    pos = [0] * (N + 1)
    for idx, val in enumerate(P):
        pos[val] = idx

    # Build array A of length N: A[i] = position of value (i+1)
    A = pos[1:]  # A[0] = pos[1], ..., A[N-1] = pos[N]

    # Deques for sliding window min and max
    min_dq = deque()
    max_dq = deque()

    ans = N  # large initial
    for i, v in enumerate(A):
        # Maintain max deque (decreasing)
        while max_dq and A[max_dq[-1]] < v:
            max_dq.pop()
        max_dq.append(i)
        # Maintain min deque (increasing)
        while min_dq and A[min_dq[-1]] > v:
            min_dq.pop()
        min_dq.append(i)

        # When window is ready
        if i >= K - 1:
            left = i - (K - 1)
            # Remove out-of-window indices
            while max_dq[0] < left:
                max_dq.popleft()
            while min_dq[0] < left:
                min_dq.popleft()
            # Current span
            curr_span = A[max_dq[0]] - A[min_dq[0]]
            if curr_span < ans:
                ans = curr_span

    # Output result
    sys.stdout.write(str(ans))

if __name__ == "__main__":
    main()