# YOUR CODE HERE
import sys
from collections import deque

def find_cycle(N, A):
    visited = [False] * (N + 1)
    parent = [-1] * (N + 1)

    def dfs(node):
        visited[node] = True
        next_node = A[node - 1]
        if not visited[next_node]:
            parent[next_node] = node
            return dfs(next_node)
        elif parent[node] != next_node:
            cycle = []
            current = node
            while current != next_node:
                cycle.append(current)
                current = parent[current]
            cycle.append(next_node)
            cycle.reverse()
            return cycle
        return None

    for i in range(1, N + 1):
        if not visited[i]:
            cycle = dfs(i)
            if cycle:
                return cycle
    return []

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    A = list(map(int, data[1:]))

    cycle = find_cycle(N, A)

    print(len(cycle))
    print(" ".join(map(str, cycle)))

if __name__ == "__main__":
    main()