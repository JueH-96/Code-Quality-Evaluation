import sys
import itertools

def read_input():
    input = sys.stdin.read
    data = input().split()

    index = 0
    N = int(data[index])
    index += 1
    M_G = int(data[index])
    index += 1
    edges_G = []
    for _ in range(M_G):
        u, v = int(data[index]), int(data[index + 1])
        index += 2
        edges_G.append((u, v))

    M_H = int(data[index])
    index += 1
    edges_H = []
    for _ in range(M_H):
        a, b = int(data[index]), int(data[index + 1])
        index += 2
        edges_H.append((a, b))

    A = []
    for i in range(1, N):
        row = []
        for j in range(i + 1, N + 1):
            row.append(int(data[index]))
            index += 1
        A.append(row)

    return N, edges_G, edges_H, A

def graph_to_adj_matrix(N, edges):
    adj_matrix = [[0] * N for _ in range(N)]
    for u, v in edges:
        adj_matrix[u-1][v-1] = 1
        adj_matrix[v-1][u-1] = 1
    return adj_matrix

def are_isomorphic(adj_matrix_G, adj_matrix_H, permutation):
    N = len(permutation)
    for i in range(N):
        for j in range(i + 1, N):
            if adj_matrix_G[i][j] != adj_matrix_H[permutation[i]-1][permutation[j]-1]:
                return False
    return True

def calculate_cost(N, edges_H, A, permutation):
    cost = 0
    for i in range(1, N):
        for j in range(i + 1, N + 1):
            if (i, j) in edges_H or (j, i) in edges_H:
                if permutation[i-1] > permutation[j-1]:
                    cost += A[permutation[j-1]-1][permutation[i-1]-1 - permutation[j-1]]
                else:
                    cost += A[permutation[i-1]-1][permutation[j-1]-1 - permutation[i-1]]
            else:
                if permutation[i-1] > permutation[j-1]:
                    cost += A[permutation[j-1]-1][permutation[i-1]-1 - permutation[j-1]]
                else:
                    cost += A[permutation[i-1]-1][permutation[j-1]-1 - permutation[i-1]]
    return cost

def main():
    N, edges_G, edges_H, A = read_input()
    adj_matrix_G = graph_to_adj_matrix(N, edges_G)
    adj_matrix_H = graph_to_adj_matrix(N, edges_H)

    min_cost = float('inf')
    for permutation in itertools.permutations(range(1, N + 1)):
        if are_isomorphic(adj_matrix_G, adj_matrix_H, permutation):
            cost = calculate_cost(N, edges_H, A, permutation)
            min_cost = min(min_cost, cost)

    print(min_cost)

if __name__ == "__main__":
    main()