def solve():
    n = int(input())
    a = list(map(int, input().split()))
    mod = 998244353
    total_sum = 0

    # Calculate the sum of f(A_i, A_j) for all i < j
    for i in range(n - 1):
        for j in range(i + 1, n):
            f_val = int(str(a[i]) + str(a[j]))
            total_sum = (total_sum + f_val) % mod

    print(total_sum)

solve()