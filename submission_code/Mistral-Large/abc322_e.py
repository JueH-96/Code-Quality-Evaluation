import itertools
import sys

def can_achieve_goal(plans, K, P):
    current_params = [0] * K
    for plan in plans:
        for j in range(K):
            current_params[j] += plan[j + 1]
    return all(param >= P for param in current_params)

def min_cost_to_achieve_goal(N, K, P, plans):
    min_cost = float('inf')
    for combination in itertools.combinations(plans, r=N):
        if can_achieve_goal(combination, K, P):
            cost = sum(plan[0] for plan in combination)
            min_cost = min(min_cost, cost)
    return min_cost if min_cost != float('inf') else -1

def main():
    input = sys.stdin.read
    data = input().split()

    index = 0
    N = int(data[index])
    K = int(data[index + 1])
    P = int(data[index + 2])
    index += 3

    plans = []
    for i in range(N):
        plan = list(map(int, data[index:index + K + 1]))
        plans.append(plan)
        index += K + 1

    result = min_cost_to_achieve_goal(N, K, P, plans)
    print(result)

if __name__ == "__main__":
    main()