def solve():
    n = int(input())
    s = input()
    total_sum = 0
    for k in range(n):
        digit = int(s[k])
        contribution = digit * (k + 1) * (10**(n - k) - 1) // 9
        total_sum += contribution
    print(total_sum)

solve()