def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    max_exp = 0
    for i in range(1 << n):
        current_exp = 0
        defeated_count = 0
        for j in range(n):
            if (i >> j) & 1:
                defeated_count += 1
                current_exp += a[j]
                if defeated_count % 2 == 0:
                    current_exp += a[j]
        max_exp = max(max_exp, current_exp)
    print(max_exp)

solve()