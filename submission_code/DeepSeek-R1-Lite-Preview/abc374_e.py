import sys
import math

def min_cost(process, W):
    A, P, B, Q = process
    min_cost = float('inf')
    # To optimize, limit the range of S to a reasonable number
    # Here, we use min(ceil(W/A), 1000000) to prevent too many iterations
    max_S = min(math.ceil(W / A), 1000000)
    for S in range(0, max_S + 1):
        remaining = W - A * S
        if remaining <= 0:
            T = 0
        else:
            T = math.ceil(remaining / B)
        cost = P * S + Q * T
        if cost < min_cost:
            min_cost = cost
    return min_cost

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    X = int(data[1])
    processes = []
    idx = 2
    for _ in range(N):
        A = int(data[idx])
        P = int(data[idx+1])
        B = int(data[idx+2])
        Q = int(data[idx+3])
        processes.append((A, P, B, Q))
        idx += 4
    # Binary search over W
    left = 0
    right = 10**12
    answer = 0
    while left <= right:
        mid = (left + right) // 2
        total_cost = 0
        for proc in processes:
            cost = min_cost(proc, mid)
            total_cost += cost
            if total_cost > X:
                break
        if total_cost <= X:
            answer = mid
            left = mid + 1
        else:
            right = mid - 1
    print(answer)

if __name__ == "__main__":
    main()