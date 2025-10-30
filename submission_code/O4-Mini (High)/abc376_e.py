import sys
import heapq

def main():
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    T = int(next(it))
    out = []
    INF = 10**30
    push = heapq.heappush
    pop = heapq.heappop

    for _ in range(T):
        N = int(next(it)); K = int(next(it))
        # read A and B
        A = [0] * N
        for i in range(N):
            A[i] = int(next(it))
        B = [0] * N
        for i in range(N):
            B[i] = int(next(it))

        # trivial cases
        if K == 1:
            # pick a single element minimizing A_i * B_i
            mn = INF
            for ai, bi in zip(A, B):
                prod = ai * bi
                if prod < mn:
                    mn = prod
            out.append(str(mn))
            continue
        if K == N:
            # pick all elements
            maxA = A[0]
            for ai in A:
                if ai > maxA:
                    maxA = ai
            sumB_all = 0
            for bi in B:
                sumB_all += bi
            out.append(str(maxA * sumB_all))
            continue

        # general case: sort by A, maintain a max-heap of the K smallest B's seen so far
        pairs = list(zip(A, B))
        pairs.sort(key=lambda x: x[0])

        heap = []
        sumB = 0
        ans = INF

        for ai, bi in pairs:
            push(heap, -bi)
            sumB += bi
            if len(heap) > K:
                # remove the largest B (i.e., pop the smallest negative)
                sumB -= -pop(heap)
            if len(heap) == K:
                cost = ai * sumB
                if cost < ans:
                    ans = cost

        out.append(str(ans))

    sys.stdout.write("
".join(out))


if __name__ == "__main__":
    main()