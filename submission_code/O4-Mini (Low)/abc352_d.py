import sys
import threading

def main():
    import sys
    from collections import deque

    input = sys.stdin.readline
    N, K = map(int, input().split())
    P = list(map(int, input().split()))
    # pos[v] = index of value v in P (1-based)
    pos = [0]*(N+1)
    for idx, val in enumerate(P, start=1):
        pos[val] = idx

    # Build array A of positions in order of values 1..N
    A = pos[1:]  # zero-based list of length N

    # We'll slide a window of size K over A, and maintain the min and max of the window
    min_deq = deque()  # will store pairs (value, index) in increasing order by value
    max_deq = deque()  # will store pairs (value, index) in decreasing order by value

    ans = N  # large initial value

    for i, x in enumerate(A):
        # Push into min_deq (maintain increasing)
        while min_deq and min_deq[-1][0] >= x:
            min_deq.pop()
        min_deq.append((x, i))

        # Push into max_deq (maintain decreasing)
        while max_deq and max_deq[-1][0] <= x:
            max_deq.pop()
        max_deq.append((x, i))

        # Once we've filled at least K items, compute window
        if i >= K-1:
            # window is [i-K+1 .. i]
            left = i - K + 1

            # Pop old elements out of window from deques
            while min_deq and min_deq[0][1] < left:
                min_deq.popleft()
            while max_deq and max_deq[0][1] < left:
                max_deq.popleft()

            cur_min = min_deq[0][0]
            cur_max = max_deq[0][0]
            # candidate answer: max position - min position
            ans = min(ans, cur_max - cur_min)

    print(ans)


if __name__ == "__main__":
    main()