def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    left = 0
    right = N // 2
    best_k = 0
    
    while left <= right:
        mid = (left + right) // 2
        possible = True
        for i in range(mid):
            if A[i] * 2 > A[N - mid + i]:
                possible = False
                break
        if possible:
            best_k = mid
            left = mid + 1
        else:
            right = mid - 1
    print(best_k)

solve()