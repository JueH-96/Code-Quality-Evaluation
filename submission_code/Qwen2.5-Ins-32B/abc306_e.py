import heapq

def solve():
    N, K, Q = map(int, input().split())
    A = [0] * N
    B = [0] * K
    B_sum = 0
    min_heap = []
    
    for _ in range(Q):
        X, Y = map(int, input().split())
        X -= 1
        
        if A[X] > 0:
            if A[X] in B:
                B[B.index(A[X])] = 0
                B_sum -= A[X]
                heapq.heapify(min_heap)
            else:
                if A[X] > min_heap[0]:
                    B_sum -= heapq.heappop(min_heap)
                    heapq.heappush(min_heap, A[X])
        
        A[X] = Y
        
        if Y > 0:
            if len(min_heap) < K:
                heapq.heappush(min_heap, Y)
                B_sum += Y
            else:
                if Y > min_heap[0]:
                    B_sum -= heapq.heappop(min_heap)
                    heapq.heappush(min_heap, Y)
                    B_sum += Y
        
        print(B_sum)

solve()