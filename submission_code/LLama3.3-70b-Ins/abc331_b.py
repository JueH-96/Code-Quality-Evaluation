import sys

def min_cost(N, S, M, L):
    min_cost = float('inf')
    for i in range(N // 6 + 1):
        for j in range(N // 8 + 1):
            remaining_eggs = N - i * 6 - j * 8
            if remaining_eggs < 0:
                continue
            if remaining_eggs % 12 == 0:
                k = remaining_eggs // 12
                total_cost = i * S + j * M + k * L
                min_cost = min(min_cost, total_cost)
            else:
                k = remaining_eggs // 12 + 1
                total_cost = i * S + j * M + k * L
                min_cost = min(min_cost, total_cost)
    return min_cost

N, S, M, L = map(int, sys.stdin.readline().split())
print(min_cost(N, S, M, L))