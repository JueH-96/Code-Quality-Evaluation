def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    K = int(input[1])
    A = list(map(int, input[2:2+K]))
    
    if K == 0:
        print(0)
        return
    
    m = K // 2
    if m == 0:
        print(0)
        return
    
    D = []
    for i in range(1, K):
        D.append(A[i] - A[i-1])
    n = len(D)
    
    take = [float('inf')] * (n + 1)
    notake = [float('inf')] * (n + 1)
    
    take[0] = float('inf')  # cannot take 0th element (since 0-based)
    notake[0] = 0
    
    if n >= 1:
        take[1] = D[0]
        notake[1] = 0
    
    for i in range(2, n + 1):
        take[i] = notake[i-1] + D[i-1]
        notake[i] = min(take[i-1], notake[i-1])
    
    res = min(take[n], notake[n])
    print(res)

if __name__ == "__main__":
    main()