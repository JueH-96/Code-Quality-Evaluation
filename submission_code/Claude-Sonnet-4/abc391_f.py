import heapq

def solve():
    N, K = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    C = list(map(int, input().split()))
    
    # Sort arrays in descending order to start with potentially largest values
    A.sort(reverse=True)
    B.sort(reverse=True)
    C.sort(reverse=True)
    
    # Use max heap (negate values for min heap)
    heap = []
    visited = set()
    
    # Start with the largest possible value
    initial_val = A[0] * B[0] + B[0] * C[0] + C[0] * A[0]
    heapq.heappush(heap, (-initial_val, 0, 0, 0))
    visited.add((0, 0, 0))
    
    for rank in range(K):
        neg_val, i, j, k = heapq.heappop(heap)
        val = -neg_val
        
        if rank == K - 1:
            print(val)
            return
        
        # Add neighboring combinations
        for di, dj, dk in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
            ni, nj, nk = i + di, j + dj, k + dk
            if (ni < N and nj < N and nk < N and 
                (ni, nj, nk) not in visited):
                new_val = A[ni] * B[nj] + B[nj] * C[nk] + C[nk] * A[ni]
                heapq.heappush(heap, (-new_val, ni, nj, nk))
                visited.add((ni, nj, nk))

solve()