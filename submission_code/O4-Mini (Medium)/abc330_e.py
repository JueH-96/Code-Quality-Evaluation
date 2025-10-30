import sys
import threading
def main():
    import sys
    data = sys.stdin
    line = data.readline().split()
    N = int(line[0]); Q = int(line[1])
    A = list(map(int, data.readline().split()))
    # We only need to track values 0..N, since mex of N-length array is at most N.
    M = N + 1
    cnt = [0] * (M)
    for v in A:
        if 0 <= v < M:
            cnt[v] += 1
    # missing[v] == True means v is currently absent from A
    missing = [False] * M
    import heapq
    heap = []
    for v in range(M):
        if cnt[v] == 0:
            missing[v] = True
            heapq.heappush(heap, v)
    out = []
    for _ in range(Q):
        parts = data.readline().split()
        i = int(parts[0]) - 1
        x = int(parts[1])
        old = A[i]
        # remove old
        if 0 <= old < M:
            cnt_old = cnt[old] - 1
            cnt[old] = cnt_old
            if cnt_old == 0:
                # now old becomes missing
                missing[old] = True
                heapq.heappush(heap, old)
        # insert new x
        A[i] = x
        if 0 <= x < M:
            cnt_x = cnt[x] + 1
            cnt[x] = cnt_x
            if cnt_x == 1:
                # x was missing, now present
                missing[x] = False
        # find mex = minimum v with missing[v]==True
        while True:
            v = heap[0]
            if missing[v]:
                out.append(str(v))
                break
            else:
                heapq.heappop(heap)
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()