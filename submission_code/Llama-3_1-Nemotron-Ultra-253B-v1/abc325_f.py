import heapq

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    D = list(map(int, data[idx:idx+N]))
    idx += N
    L1, C1, K1 = map(int, data[idx:idx+3])
    idx +=3
    L2, C2, K2 = map(int, data[idx:idx+3])
    
    INF = float('inf')
    visited = [[[INF] * (K2 + 1) for _ in range(K1 + 1)] for __ in range(N + 1)]
    visited[0][0][0] = 0
    heap = []
    heapq.heappush(heap, (0, 0, 0, 0))
    
    while heap:
        cost, i, a, b = heapq.heappop(heap)
        if i == N:
            print(cost)
            return
        if cost > visited[i][a][b]:
            continue
        current_D = D[i]
        max_x1 = K1 - a
        max_x2 = K2 - b
        for x1 in range(0, max_x1 + 1):
            if x1 * L1 >= current_D:
                x2 = 0
            else:
                remaining = current_D - x1 * L1
                x2 = (remaining + L2 - 1) // L2
            if x2 > max_x2:
                continue
            new_a = a + x1
            new_b = b + x2
            new_cost = cost + x1 * C1 + x2 * C2
            if new_a > K1 or new_b > K2:
                continue
            if new_cost < visited[i+1][new_a][new_b]:
                visited[i+1][new_a][new_b] = new_cost
                heapq.heappush(heap, (new_cost, i+1, new_a, new_b))
    
    print(-1)

if __name__ == '__main__':
    main()