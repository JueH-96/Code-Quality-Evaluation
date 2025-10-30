import sys
import itertools

def read_input():
    N, K, P = map(int, sys.stdin.readline().split())
    plans = []
    for _ in range(N):
        plan = list(map(int, sys.stdin.readline().split()))
        plans.append(plan)
    return N, K, P, plans

def calculate_cost(N, K, P, plans):
    min_cost = float('inf')
    for r in range(1, N + 1):
        for combination in itertools.combinations(range(N), r):
            total_cost = 0
            parameter_values = [0] * K
            for i in combination:
                total_cost += plans[i][0]
                for j in range(1, K + 1):
                    parameter_values[j - 1] += plans[i][j]
            if all(value >= P for value in parameter_values):
                min_cost = min(min_cost, total_cost)
    return min_cost if min_cost != float('inf') else -1

def main():
    N, K, P, plans = read_input()
    min_cost = calculate_cost(N, K, P, plans)
    print(min_cost)

if __name__ == "__main__":
    main()