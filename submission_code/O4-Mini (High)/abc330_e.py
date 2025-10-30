import sys
import heapq

def main():
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    N = int(next(it)); Q = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    M = N  # We only need to track 0..N for the mex

    # Count occurrences of each v in [0..M]
    cnt = [0] * (M + 1)
    for v in A:
        if v <= M:
            cnt[v] += 1

    # is_missing[x] = True iff x is currently not in A
    is_missing = [False] * (M + 1)
    heap = []
    for x in range(M + 1):
        if cnt[x] == 0:
            is_missing[x] = True
            heap.append(x)
    heapq.heapify(heap)

    out = []
    for _ in range(Q):
        idx = int(next(it)) - 1
        x = int(next(it))

        old = A[idx]
        # Remove the old value
        if old <= M:
            cnt_old = cnt[old] - 1
            cnt[old] = cnt_old
            if cnt_old == 0:
                # old value became missing
                is_missing[old] = True
                heapq.heappush(heap, old)

        # Insert the new value
        A[idx] = x
        if x <= M:
            if cnt[x] == 0:
                # x was missing, now becomes present
                is_missing[x] = False
            cnt[x] += 1

        # Find current mex: the smallest x with is_missing[x] == True
        while True:
            top = heap[0]
            if is_missing[top]:
                out.append(str(top))
                break
            else:
                heapq.heappop(heap)

    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()