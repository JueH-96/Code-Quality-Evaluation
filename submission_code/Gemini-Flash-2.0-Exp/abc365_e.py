def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    total_xor_sum = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            xor_sum = 0
            for k in range(i, j + 1):
                xor_sum ^= a[k]
            total_xor_sum += xor_sum
    
    print(total_xor_sum)

solve()