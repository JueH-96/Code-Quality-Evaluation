import sys
import threading

def main():
    import sys
    data = sys.stdin
    T = int(data.readline())
    out = []
    for _ in range(T):
        line = data.readline().split()
        while len(line) < 2:
            line += data.readline().split()
        N = int(line[0]); K = int(line[1])
        A = list(map(int, data.readline().split()))
        B = list(map(int, data.readline().split()))
        # Special case K == 1: pick single i minimizing A[i]*B[i]
        if K == 1:
            ans = min(A[i] * B[i] for i in range(N))
            out.append(str(ans))
            continue
        # General K >= 2
        items = list(zip(A, B))
        items.sort(key=lambda x: x[0])  # sort by A ascending
        import heapq
        maxheap = []   # will store negative B's, so the largest B is on top
        sumB = 0
        best = 10**30
        for a, b in items:
            # push b
            heapq.heappush(maxheap, -b)
            sumB += b
            # if we have more than K, pop the largest B (i.e. most negative of -b)
            if len(maxheap) > K:
                popped = -heapq.heappop(maxheap)
                sumB -= popped
            # if exactly K, evaluate candidate
            if len(maxheap) == K:
                # max A is 'a', sum of chosen B's is sumB
                cand = a * sumB
                if cand < best:
                    best = cand
        out.append(str(best))
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()