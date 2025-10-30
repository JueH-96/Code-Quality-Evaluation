import heapq

def find_min_cost(N, D, P, fares):
    total_cost = 0
    passes_used = 0
    fares_heap = []
    daily_fares = 0
    
    for i in range(N):
        daily_fares += fares[i]
        heapq.heappush(fares_heap, -fares[i])
        
        if (i + 1) % D == 0 or i == N - 1:
            while passes_used < (i + 1) // D:
                total_cost += P
                passes_used += D
                if fares_heap:
                    daily_fares += heapq.heappop(fares_heap)
            total_cost += daily_fares
            daily_fares = 0
    
    return total_cost

N, D, P = map(int, input().split())
fares = list(map(int, input().split()))

print(find_min_cost(N, D, P, fares))