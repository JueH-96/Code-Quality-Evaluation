import sys
import threading
def main():
    import sys
    import heapq

    input = sys.stdin.readline
    N, K, Q = map(int, input().split())
    # A: current values
    A = [0] * N
    # which[i]: 1 if in top-K heap (H1), 2 if in rest heap (H2), 0 as temp
    which = [0] * N

    # Min-heap for S1 (top K): stores (value, index)
    H1 = []
    # Max-heap for S2 (rest): stores (-value, index)
    H2 = []

    sum1 = 0  # sum of values in H1
    len1 = 0  # number of elements in H1

    # Initialize: first K indices into H1, rest into H2
    for i in range(N):
        if i < K:
            which[i] = 1
            len1 += 1
            heapq.heappush(H1, (0, i))
        else:
            which[i] = 2
            heapq.heappush(H2, (0, i))  # store as ( -0, i ) but 0 is same
    # Helpers to clean stale tops
    def clean_H1():
        # pop until top is valid
        while H1:
            v, i = H1[0]
            # valid iff which[i]==1 and A[i]==v
            if which[i] != 1 or A[i] != v:
                heapq.heappop(H1)
            else:
                break

    def clean_H2():
        while H2:
            nv, i = H2[0]
            v = -nv
            if which[i] != 2 or A[i] != v:
                heapq.heappop(H2)
            else:
                break

    out = []
    for _ in range(Q):
        line = input().split()
        if not line:
            line = input().split()
        x = int(line[0]) - 1
        y = int(line[1])
        old = A[x]
        w = which[x]
        # Remove old
        if w == 1:
            sum1 -= old
            len1 -= 1
        which[x] = 0
        A[x] = y
        # Decide where to insert new
        clean_H1()
        if len1 < K:
            # still room in top-K
            which[x] = 1
            len1 += 1
            sum1 += y
            heapq.heappush(H1, (y, x))
        else:
            # compare with smallest in H1
            # ensure H1 top is clean
            if H1:
                min1 = H1[0][0]
            else:
                min1 = -1  # no elements
            if y > min1:
                which[x] = 1
                len1 += 1
                sum1 += y
                heapq.heappush(H1, (y, x))
            else:
                which[x] = 2
                heapq.heappush(H2, (-y, x))
        # Rebalance to maintain len1 == K
        if len1 > K:
            # move smallest from H1 to H2
            clean_H1()
            v, i = heapq.heappop(H1)
            which[i] = 2
            len1 -= 1
            sum1 -= v
            heapq.heappush(H2, (-v, i))
        elif len1 < K:
            # move largest from H2 to H1
            clean_H2()
            nv, i = heapq.heappop(H2)
            v = -nv
            which[i] = 1
            len1 += 1
            sum1 += v
            heapq.heappush(H1, (v, i))
        out.append(str(sum1))

    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()