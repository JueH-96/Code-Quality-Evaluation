import sys
import threading

def main():
    import sys
    import heapq

    input = sys.stdin.readline
    N, K = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    C = list(map(int, input().split()))

    # Sort in descending order so the largest sums come first
    A.sort(reverse=True)
    B.sort(reverse=True)
    C.sort(reverse=True)

    # Max-heap: store (-value, i, j, k)
    hq = []
    # Visited set of (i,j,k) to avoid duplicates
    visited = set()

    # Initial state: indices (0,0,0)
    init_val = A[0]*B[0] + B[0]*C[0] + C[0]*A[0]
    heapq.heappush(hq, (-init_val, 0, 0, 0))
    visited.add((0,0,0))

    ans = None
    # Extract the top K states
    for _ in range(K):
        neg_val, i, j, k = heapq.heappop(hq)
        ans = -neg_val
        # Push neighbors (i+1,j,k), (i,j+1,k), (i,j,k+1)
        if i+1 < N:
            t = (i+1, j, k)
            if t not in visited:
                visited.add(t)
                v = A[i+1]*B[j] + B[j]*C[k] + C[k]*A[i+1]
                heapq.heappush(hq, (-v, i+1, j, k))
        if j+1 < N:
            t = (i, j+1, k)
            if t not in visited:
                visited.add(t)
                v = A[i]*B[j+1] + B[j+1]*C[k] + C[k]*A[i]
                heapq.heappush(hq, (-v, i, j+1, k))
        if k+1 < N:
            t = (i, j, k+1)
            if t not in visited:
                visited.add(t)
                v = A[i]*B[j] + B[j]*C[k+1] + C[k+1]*A[i]
                heapq.heappush(hq, (-v, i, j, k+1))

    print(ans)

if __name__ == "__main__":
    main()