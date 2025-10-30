def main():
    import sys
    N, D, P = map(int, sys.stdin.readline().split())
    F = list(map(int, sys.stdin.readline().split()))
    F.sort(reverse=True)
    prefix = [0]
    for f in F:
        prefix.append(prefix[-1] + f)
    total_sum = prefix[-1]
    K_max = (N + D - 1) // D
    min_cost = float('inf')
    for K in range(0, K_max + 1):
        days = K * D
        idx = min(days, N)
        sum_c = prefix[idx]
        cost = K * P + (total_sum - sum_c)
        if cost < min_cost:
            min_cost = cost
    print(min_cost)

if __name__ == "__main__":
    main()