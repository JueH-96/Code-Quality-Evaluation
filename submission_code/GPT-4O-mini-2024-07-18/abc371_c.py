from itertools import permutations

def read_input():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    index = 0
    N = int(data[index])
    index += 1
    
    M_G = int(data[index])
    index += 1
    
    edges_G = []
    for _ in range(M_G):
        u, v = map(int, data[index].split())
        edges_G.append((u - 1, v - 1))  # Convert to 0-indexed
        index += 1
    
    M_H = int(data[index])
    index += 1
    
    edges_H = []
    for _ in range(M_H):
        a, b = map(int, data[index].split())
        edges_H.append((a - 1, b - 1))  # Convert to 0-indexed
        index += 1
    
    costs = []
    for i in range(1, N):
        costs.append(list(map(int, data[index].split())))
        index += 1
    
    return N, edges_G, edges_H, costs

def build_edge_set(edges):
    return set(edges)

def calculate_cost(permutation, edges_G, edges_H, costs):
    edge_set_G = build_edge_set(edges_G)
    edge_set_H = build_edge_set((permutation[u], permutation[v]) for u, v in edges_H)
    
    total_cost = 0
    
    for i in range(N):
        for j in range(i + 1, N):
            if (i, j) in edge_set_G:
                if (permutation[i], permutation[j]) not in edge_set_H:
                    total_cost += costs[i][j - i - 1]  # Cost to add edge
            else:
                if (permutation[i], permutation[j]) in edge_set_H:
                    total_cost += costs[i][j - i - 1]  # Cost to remove edge
    
    return total_cost

def find_minimum_cost(N, edges_G, edges_H, costs):
    min_cost = float('inf')
    
    for perm in permutations(range(N)):
        cost = calculate_cost(perm, edges_G, edges_H, costs)
        min_cost = min(min_cost, cost)
    
    return min_cost

def main():
    N, edges_G, edges_H, costs = read_input()
    result = find_minimum_cost(N, edges_G, edges_H, costs)
    print(result)

if __name__ == "__main__":
    main()