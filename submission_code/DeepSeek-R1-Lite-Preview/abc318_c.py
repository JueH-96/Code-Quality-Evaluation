def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    D = int(data[1])
    P = int(data[2])
    F = list(map(int, data[3:3+N]))
    
    F.sort()
    prefix_sum = [0] * (N + 1)
    for i in range(N):
        prefix_sum[i+1] = prefix_sum[i] + F[i]
    
    max_k = (N + D - 1) // D
    min_cost = float('inf')
    
    for k in range(0, max_k + 1):
        passes = k * D
        remaining_days = N - passes
        if remaining_days < 0:
            remaining_days = 0
        if remaining_days > 0:
            total_cost = k * P + prefix_sum[remaining_days]
        else:
            total_cost = k * P
        if total_cost < min_cost:
            min_cost = total_cost
    
    print(min_cost)

if __name__ == '__main__':
    main()