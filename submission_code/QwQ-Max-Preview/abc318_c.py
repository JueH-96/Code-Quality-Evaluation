def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx]); idx += 1
    D = int(input[idx]); idx += 1
    P = int(input[idx]); idx += 1
    F = list(map(int, input[idx:idx+N]))
    
    F.sort(reverse=True)
    prefix = [0] * (N + 1)
    for i in range(N):
        prefix[i+1] = prefix[i] + F[i]
    total_sum = prefix[N]
    K_max = (N + D - 1) // D
    min_cost = float('inf')
    
    for K in range(K_max + 1):
        x = K * D
        x = min(x, N)
        sum_covered = prefix[x]
        cost = K * P + (total_sum - sum_covered)
        if cost < min_cost:
            min_cost = cost
    
    print(min_cost)

if __name__ == "__main__":
    main()