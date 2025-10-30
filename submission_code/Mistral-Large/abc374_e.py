import sys
input = sys.stdin.read

def max_production_capacity(N, X, processes):
    left, right = 0, 10**12

    while right - left > 1:
        mid = (left + right) // 2
        if can_achieve_capacity(N, X, processes, mid):
            left = mid
        else:
            right = mid

    return left

def can_achieve_capacity(N, X, processes, capacity):
    total_cost = 0

    for A, P, B, Q in processes:
        if capacity <= 0:
            continue
        cost_S = (capacity + A - 1) // A * P
        cost_T = (capacity + B - 1) // B * Q
        total_cost += min(cost_S, cost_T)

        if total_cost > X:
            return False

    return True

def main():
    data = input().split()
    index = 0
    N = int(data[index])
    index += 1
    X = int(data[index])
    index += 1
    processes = []

    for _ in range(N):
        A = int(data[index])
        index += 1
        P = int(data[index])
        index += 1
        B = int(data[index])
        index += 1
        Q = int(data[index])
        index += 1
        processes.append((A, P, B, Q))

    result = max_production_capacity(N, X, processes)
    print(result)

if __name__ == "__main__":
    main()