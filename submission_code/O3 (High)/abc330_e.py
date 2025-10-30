import sys
import heapq

def main() -> None:
    input_data = sys.stdin.buffer.read().split()
    it = iter(input_data)

    N = int(next(it))
    Q = int(next(it))

    # Current array
    A = [int(next(it)) for _ in range(N)]

    # We only need frequencies for numbers in 0 … N  (mex ≤ N always)
    freq = [0] * (N + 2)          # size N+2 is enough to access index N safely

    for v in A:
        if v <= N:
            freq[v] += 1

    # min-heap that stores all values whose current frequency is 0
    zero_heap = []
    for v in range(N + 1):
        if freq[v] == 0:
            heapq.heappush(zero_heap, v)

    out_lines = []

    for _ in range(Q):
        idx = int(next(it)) - 1   # convert to 0-based
        x   = int(next(it))

        old = A[idx]

        # remove old value
        if old <= N:
            freq[old] -= 1
            if freq[old] == 0:
                heapq.heappush(zero_heap, old)

        # insert new value
        A[idx] = x
        if x <= N:
            if freq[x] == 0:          # becomes positive, but we keep the zero in heap (lazy deletion)
                pass
            freq[x] += 1

        # obtain current mex
        while True:
            cand = zero_heap[0]
            if freq[cand] == 0:
                out_lines.append(str(cand))
                break
            heapq.heappop(zero_heap)   # outdated entry, discard

    sys.stdout.write('
'.join(out_lines))

if __name__ == "__main__":
    main()