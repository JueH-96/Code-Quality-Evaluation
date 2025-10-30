def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    D = int(data[1])
    P = int(data[2])
    F = list(map(int, data[3:3+N]))
    
    F.sort(reverse=True)
    
    prefix = [0] * (N + 1)
    for i in range(N):
        prefix[i+1] = prefix[i] + F[i]
    total = prefix[N]
    
    K_upper = (N + D - 1) // D
    min_cost = total  # cost when K=0
    
    for K in range(K_upper + 1):
        x = K * D
        if x <= N:
            saved = prefix[x]
        else:
            saved = total
        cost = K * P + (total - saved)
        if cost < min_cost:
            min_cost = cost
    
    print(min_cost)

if __name__ == '__main__':
    main()