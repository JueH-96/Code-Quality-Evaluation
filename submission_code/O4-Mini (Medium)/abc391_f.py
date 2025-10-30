import sys
import threading
def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    K = int(next(it))
    A = [int(next(it)) for _ in range(n)]
    B = [int(next(it)) for _ in range(n)]
    C = [int(next(it)) for _ in range(n)]
    # sort descending
    A.sort(reverse=True)
    B.sort(reverse=True)
    C.sort(reverse=True)
    b0 = B[0]
    import heapq
    # Step 1: find top L = min(K, n*n) pairs (i,k) by f(i,k,0) = A[i]*b0 + b0*C[k] + A[i]*C[k]
    L = K if K <= n*n else n*n
    heap2d = []
    visited2d = set()
    # initial (0,0)
    v00 = A[0]*b0 + b0*C[0] + A[0]*C[0]
    heapq.heappush(heap2d, (-v00, 0, 0))
    visited2d.add(0)  # key = i*n + k
    pairs = []  # will store tuples (i, k, val0)
    push2d = heapq.heappush
    pop2d = heapq.heappop
    # BFS on 2D
    cnt2d = 0
    while cnt2d < L and heap2d:
        negv, i, k = pop2d(heap2d)
        val = -negv
        pairs.append((i, k, val))
        cnt2d += 1
        # neighbor (i+1, k)
        ni = i + 1
        if ni < n:
            key = ni * n + k
            if key not in visited2d:
                visited2d.add(key)
                v1 = A[ni]*b0 + b0*C[k] + A[ni]*C[k]
                push2d(heap2d, (-v1, ni, k))
        # neighbor (i, k+1)
        nk = k + 1
        if nk < n:
            key = i * n + nk
            if key not in visited2d:
                visited2d.add(key)
                v2 = A[i]*b0 + b0*C[nk] + A[i]*C[nk]
                push2d(heap2d, (-v2, i, nk))
    # Now we have L = len(pairs) sequences, each sequence over j=0..n-1:
    # f(i,j,k) = (A[i]+C[k]) * B[j] + A[i]*C[k]
    L = len(pairs)
    ai_ck = [0] * L
    s0 = [0] * L
    for idx in range(L):
        i, k, _ = pairs[idx]
        ai = A[i]
        ck = C[k]
        ai_ck[idx] = ai + ck
        s0[idx] = ai * ck
    # Step 2: k-way merge over j
    heap1d = []
    push1d = heapq.heappush
    pop1d = heapq.heappop
    # initial for each sequence j=0, val = pairs[idx][2]
    for idx in range(L):
        val0 = pairs[idx][2]
        push1d(heap1d, (-val0, idx, 0))
    # extract K-th
    ans = None
    cnt = 0
    while cnt < K and heap1d:
        negv, seq, j = pop1d(heap1d)
        val = -negv
        cnt += 1
        if cnt == K:
            ans = val
            break
        nj = j + 1
        if nj < n:
            # compute next value
            vnext = ai_ck[seq] * B[nj] + s0[seq]
            push1d(heap1d, (-vnext, seq, nj))
    # print answer
    sys.stdout.write(str(ans))

if __name__ == "__main__":
    main()