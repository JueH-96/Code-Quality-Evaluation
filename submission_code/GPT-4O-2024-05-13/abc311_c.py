# YOUR CODE HERE
def find_cycle(N, A):
    visited = [False] * (N + 1)
    stack = []
    in_stack = [False] * (N + 1)
    
    def dfs(v):
        visited[v] = True
        stack.append(v)
        in_stack[v] = True
        
        next_v = A[v - 1]
        if not visited[next_v]:
            if dfs(next_v):
                return True
        elif in_stack[next_v]:
            cycle_start = next_v
            cycle = []
            while stack:
                node = stack.pop()
                cycle.append(node)
                in_stack[node] = False
                if node == cycle_start:
                    break
            cycle.reverse()
            print(len(cycle))
            print(" ".join(map(str, cycle)))
            return True
        
        stack.pop()
        in_stack[v] = False
        return False
    
    for i in range(1, N + 1):
        if not visited[i]:
            if dfs(i):
                break

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
A = list(map(int, data[1:]))
find_cycle(N, A)