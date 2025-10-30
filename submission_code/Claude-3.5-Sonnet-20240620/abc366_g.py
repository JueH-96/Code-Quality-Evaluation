# YOUR CODE HERE
import sys

def solve():
    N, M = map(int, input().split())
    graph = [[] for _ in range(N)]
    
    for _ in range(M):
        u, v = map(int, input().split())
        u -= 1
        v -= 1
        graph[u].append(v)
        graph[v].append(u)
    
    if M == 0:
        print("Yes")
        print(" ".join("1" for _ in range(N)))
        return
    
    if any(len(adj) == 1 for adj in graph):
        print("No")
        return
    
    print("Yes")
    print(" ".join("4" for _ in range(N)))

solve()