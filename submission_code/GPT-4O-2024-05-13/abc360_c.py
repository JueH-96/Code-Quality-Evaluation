# YOUR CODE HERE
import sys
input = sys.stdin.read

def find_min_cost(N, A, W):
    from collections import defaultdict, deque
    
    # Create a graph of the item movements
    graph = defaultdict(list)
    for i in range(N):
        graph[A[i] - 1].append(i)
    
    visited = [False] * N
    min_cost = 0
    
    def find_cycle(start):
        cycle = []
        queue = deque([start])
        while queue:
            node = queue.pop()
            if visited[node]:
                continue
            visited[node] = True
            cycle.append(node)
            for neighbor in graph[node]:
                if not visited[neighbor]:
                    queue.append(neighbor)
        return cycle
    
    for i in range(N):
        if not visited[i]:
            cycle = find_cycle(i)
            if len(cycle) > 1:
                cycle_weights = [W[item] for item in cycle]
                min_weight = min(cycle_weights)
                cycle_sum = sum(cycle_weights)
                min_cost += cycle_sum - min_weight
    
    return min_cost

def main():
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    W = list(map(int, data[N+1:2*N+1]))
    
    result = find_min_cost(N, A, W)
    print(result)

if __name__ == "__main__":
    main()