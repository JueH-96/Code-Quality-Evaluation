def solve():
    N = int(input())
    A = list(map(int, input().split()))
    
    result = 0
    for j in range(N):
        count = 0
        sum_less = 0
        for i in range(j):
            if A[i] < A[j]:
                count += 1
                sum_less += A[i]
        result += A[j] * count - sum_less
    
    print(result)

solve()