import sys
import heapq

def main():
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    N = int(data[index])
    index += 1
    M = int(data[index])
    index += 1
    
    graph = [[] for _ in range(N)]
    
    for _ in range(M):
        u = int(data[index]) - 1
        index += 1
        v = int(data[index]) - 1
        index += 1
        graph[u].append(v)
        graph[v].append(u)
    
    W = []
    for _ in range(N):
        W.append(int(data[index]))
        index += 1
    
    A = []
    for _ in range(N):
        A.append(int(data[index]))
        index += 1
    
    # Priority queue to process vertices with pieces
    pq = []
    for i in range(N):
        if A[i] > 0:
            heapq.heappush(pq, (-W[i], i))
    
    operations = 0
    
    while pq:
        _, x = heapq.heappop(pq)
        
        if A[x] == 0:
            continue
        
        # Remove one piece from x
        A[x] -= 1
        operations += 1
        
        # Find the best set S of neighbors to place pieces on
        neighbors = graph[x]
        neighbors.sort(key=lambda y: W[y])
        
        S = []
        sum_weights = 0
        for y in neighbors:
            if sum_weights + W[y] < W[x]:
                S.append(y)
                sum_weights += W[y]
            else:
                break
        
        # Place one piece on each vertex in S
        for y in S:
            A[y] += 1
            if A[y] == 1:
                heapq.heappush(pq, (-W[y], y))
    
    print(operations)