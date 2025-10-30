# YOUR CODE HERE
from collections import defaultdict
import heapq

def main():
    N, Q = map(int, input().split())
    graph = defaultdict(set)

    for _ in range(Q):
        query = list(map(int, input().split()))
        
        if query[0] == 1:
            _, u, v = query
            graph[u].add(v)
            graph[v].add(u)
        
        elif query[0] == 2:
            _, v, k = query
            connected = list(graph[v])
            connected.append(v)
            
            if len(connected) < k:
                print(-1)
            else:
                largest_k = heapq.nlargest(k, connected)
                print(largest_k[-1])

if __name__ == "__main__":
    main()