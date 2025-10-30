def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    total_xor_sum = 0
    for i in range(n):
        current_xor = 0
        for j in range(i, n):
            current_xor ^= a[j]
            if i < j:
                total_xor_sum += current_xor
    print(total_xor_sum)

solve()