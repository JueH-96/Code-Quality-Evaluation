import heapq

def main():
    N, K, Q = map(int, input().split())
    A = [0] * N
    top_K = []
    rest = []
    
    def update_f(A, x, y):
        old_val = A[x-1]
        A[x-1] = y
        
        if old_val in top_K:
            top_K.remove(old_val)
            heapq.heapify(top_K)
            rest.append(old_val)
        
        if len(top_K) < K or (rest and y > rest[0]):
            if rest:
                heapq.heappush(top_K, y)
                if len(top_K) > K:
                    heapq.heappush(rest, heapq.heappop(top_K))
            else:
                heapq.heappush(top_K, y)
        else:
            heapq.heappush(rest, y)
        
        return sum(top_K)
    
    for _ in range(Q):
        x, y = map(int, input().split())
        print(update_f(A, x, y))

if __name__ == "__main__":
    main()