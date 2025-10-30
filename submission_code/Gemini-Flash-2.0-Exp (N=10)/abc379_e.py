def solve():
    n = int(input())
    s = input()
    total_sum = 0
    for i in range(n):
        for j in range(i, n):
            total_sum += int(s[i:j+1])
    print(total_sum)

solve()