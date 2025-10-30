import sys
from collections import deque

def main() -> None:
    input = sys.stdin.readline

    N, K = map(int, input().split())
    P = list(map(int, input().split()))

    # position[value] = index of that value in the permutation (1-based)
    position = [0] * (N + 1)
    for idx, val in enumerate(P, 1):
        position[val] = idx

    # Trivial case: picking a single element always yields span 0
    if K == 1:
        print(0)
        return

    # Array of positions for values 1..N
    A = position[1:]                     # length N, 0-based indices

    # Monotone deques to keep current window minimum and maximum
    dq_min = deque()                     # increasing by value
    dq_max = deque()                     # decreasing by value
    best = N                             # span is at most N-1

    for i, v in enumerate(A):
        # Maintain deque for minima
        while dq_min and dq_min[-1][0] >= v:
            dq_min.pop()
        dq_min.append((v, i))

        # Maintain deque for maxima
        while dq_max and dq_max[-1][0] <= v:
            dq_max.pop()
        dq_max.append((v, i))

        # When we have a full window [i-K+1, i]
        if i >= K - 1:
            left = i - K + 1

            # Discard elements that left the window
            while dq_min[0][1] < left:
                dq_min.popleft()
            while dq_max[0][1] < left:
                dq_max.popleft()

            # Update answer
            span = dq_max[0][0] - dq_min[0][0]
            if span < best:
                best = span

    print(best)

if __name__ == "__main__":
    main()