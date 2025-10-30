import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(10**7)
    data = sys.stdin.read().split()
    n = int(data[0])
    A = [0] + list(map(int, data[1:]))

    if n < 2:
        print(0)
        return

    # doubly linked list via arrays
    left = [i-1 for i in range(n+1)]
    right = [i+1 for i in range(n+1)]
    left[1] = 0
    right[n] = 0

    alive = [True]*(n+1)

    import heapq
    heap = []
    # push all initial adjacent pairs
    for i in range(1, n):
        w = abs(A[i] - A[i+1])
        # we store negative weight for max-heap
        heapq.heappush(heap, (-w, i, i+1))

    total = 0
    # process removals
    while heap:
        negw, l, r = heapq.heappop(heap)
        # check still valid: both alive and adjacent
        if not alive[l] or not alive[r]:
            continue
        if right[l] != r or left[r] != l:
            continue
        # take this removal
        w = -negw
        total += w
        # remove l and r
        alive[l] = False
        alive[r] = False
        L = left[l]
        R = right[r]
        # link L and R
        if L:
            right[L] = R
        if R:
            left[R] = L
        # if new adjacent, push their diff
        if L and R:
            ww = abs(A[L] - A[R])
            heapq.heappush(heap, (-ww, L, R))

    print(total)

if __name__ == "__main__":
    main()