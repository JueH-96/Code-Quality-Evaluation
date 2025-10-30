import sys
import threading
import heapq

def main():
    import sys
    sys.setrecursionlimit(10**7)
    input = sys.stdin.readline

    T = int(input())
    for _ in range(T):
        line = input().split()
        while not line:
            line = input().split()
        N, K = map(int, line)
        X = [0]*N
        Y = [0]*N
        Z = [0]*N
        for i in range(N):
            xi, yi, zi = map(int, input().split())
            X[i]=xi; Y[i]=yi; Z[i]=zi

        # Three max-heaps for X, Y, Z
        hX = [(-X[i], i) for i in range(N)]
        hY = [(-Y[i], i) for i in range(N)]
        hZ = [(-Z[i], i) for i in range(N)]
        heapq.heapify(hX)
        heapq.heapify(hY)
        heapq.heapify(hZ)
        heaps = [hX, hY, hZ]
        vals = [X, Y, Z]

        removed = [False]*N

        # For a node u and dimension d, find the top node v≠u, not removed.
        # We pop invalid entries and then push them back.
        def best_in_dim(u, d):
            h = heaps[d]
            buf = []
            v = -1
            # pop until we find a valid v≠u
            while h:
                negval, cand = h[0]
                if removed[cand] or cand == u:
                    # invalid, pop it off
                    heapq.heappop(h)
                    buf.append((negval, cand))
                else:
                    v = cand
                    break
            # restore popped entries
            for item in buf:
                heapq.heappush(h, item)
            return v

        # For node u, find its best partner v and the corresponding weight w
        def get_best(u):
            best_w = -1
            best_v = -1
            # try each dimension
            for d in (0,1,2):
                v = best_in_dim(u, d)
                if v<0: 
                    continue
                w = vals[d][u] + vals[d][v]
                if w > best_w:
                    best_w = w
                    best_v = v
            return best_v, best_w

        # Global max-heap of candidate edges (by negative weight)
        pq = []
        for u in range(N):
            v, w = get_best(u)
            if v>=0:
                heapq.heappush(pq, (-w, u, v))

        total = 0
        matches = 0
        # greedy extract the best edge, lazily check validity
        while matches < K and pq:
            negw, u, v = heapq.heappop(pq)
            w = -negw
            if removed[u] or removed[v]:
                continue
            # re-check that v is still u's best partner and weight matches
            v0, w0 = get_best(u)
            if v0!=v or w0!=w:
                # outdated, push the updated best for u
                if v0>=0:
                    heapq.heappush(pq, (-w0, u, v0))
                continue
            # accept this match
            removed[u] = True
            removed[v] = True
            total += w
            matches += 1

        print(total)

if __name__=="__main__":
    main()