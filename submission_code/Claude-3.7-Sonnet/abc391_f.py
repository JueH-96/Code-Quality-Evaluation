import heapq

def main():
    N, K = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    C = list(map(int, input().split()))
    
    min_heap = []  # Min heap to keep the K largest values
    
    for i in range(N):
        for j in range(N):
            for k in range(N):
                value = A[i] * B[j] + B[j] * C[k] + C[k] * A[i]
                
                if len(min_heap) < K:
                    heapq.heappush(min_heap, value)
                elif value > min_heap[0]:
                    heapq.heappushpop(min_heap, value)
    
    print(min_heap[0])

if __name__ == "__main__":
    main()