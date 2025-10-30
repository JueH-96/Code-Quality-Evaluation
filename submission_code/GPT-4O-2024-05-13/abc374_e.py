# YOUR CODE HERE
import sys
input = sys.stdin.read

def can_achieve_capacity(N, X, processes, target_capacity):
    total_cost = 0
    for A, P, B, Q in processes:
        min_cost = float('inf')
        for x in range(target_capacity // A + 1):
            y = (target_capacity - x * A + B - 1) // B
            if y < 0:
                continue
            cost = x * P + y * Q
            min_cost = min(min_cost, cost)
        total_cost += min_cost
        if total_cost > X:
            return False
    return total_cost <= X

def max_production_capacity(N, X, processes):
    low, high = 0, 10**9
    while low < high:
        mid = (low + high + 1) // 2
        if can_achieve_capacity(N, X, processes, mid):
            low = mid
        else:
            high = mid - 1
    return low

def main():
    data = input().strip().split()
    N = int(data[0])
    X = int(data[1])
    processes = []
    index = 2
    for _ in range(N):
        A = int(data[index])
        P = int(data[index + 1])
        B = int(data[index + 2])
        Q = int(data[index + 3])
        processes.append((A, P, B, Q))
        index += 4
    result = max_production_capacity(N, X, processes)
    print(result)

if __name__ == "__main__":
    main()