import sys
import heapq

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    
    A = list(map(int, data[2:N+2]))
    B = list(map(int, data[N+2:2*N+2]))
    C = list(map(int, data[2*N+2:3*N+2]))
    
    # We will use a max-heap to keep track of the K-th largest elements
    max_heap = []
    
    # Compute all possible values of A_i * B_j + B_j * C_k + C_k * A_i
    for i in range(N):
        for j in range(N):
            for k in range(N):
                value = A[i] * B[j] + B[j] * C[k] + C[k] * A[i]
                # Use a min-heap by pushing negative values
                if len(max_heap) < K:
                    heapq.heappush(max_heap, value)
                else:
                    # Only push if the current value is larger than the smallest in the heap
                    if value > max_heap[0]:
                        heapq.heappushpop(max_heap, value)
    
    # The K-th largest element is the root of the heap
    print(max_heap[0])

if __name__ == "__main__":
    main()