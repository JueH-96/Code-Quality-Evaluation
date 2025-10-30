import heapq
import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:2+N]))
    B = list(map(int, data[2+N:2+2*N]))
    C = list(map(int, data[2+2*N:2+3*N]))
    
    # Sort A, B, C in descending order
    A.sort(reverse=True)
    B.sort(reverse=True)
    C.sort(reverse=True)
    
    # Max-heap to store the top K largest values
    max_heap = []
    
    # We will use a set to avoid duplicate entries in the heap
    visited = set()
    
    # Start with the largest possible combination
    heapq.heappush(max_heap, (-(A[0] * B[0] + B[0] * C[0] + C[0] * A[0]), 0, 0, 0))
    visited.add((0, 0, 0))
    
    # Extract the K-th largest
    for _ in range(K):
        current_value, i, j, k = heapq.heappop(max_heap)
        current_value = -current_value
        
        # Try to push the next possible combinations into the heap
        if i + 1 < N and (i + 1, j, k) not in visited:
            heapq.heappush(max_heap, (-(A[i + 1] * B[j] + B[j] * C[k] + C[k] * A[i + 1]), i + 1, j, k))
            visited.add((i + 1, j, k))
        
        if j + 1 < N and (i, j + 1, k) not in visited:
            heapq.heappush(max_heap, (-(A[i] * B[j + 1] + B[j + 1] * C[k] + C[k] * A[i]), i, j + 1, k))
            visited.add((i, j + 1, k))
        
        if k + 1 < N and (i, j, k + 1) not in visited:
            heapq.heappush(max_heap, (-(A[i] * B[j] + B[j] * C[k + 1] + C[k + 1] * A[i]), i, j, k + 1))
            visited.add((i, j, k + 1))
    
    print(current_value)

main()