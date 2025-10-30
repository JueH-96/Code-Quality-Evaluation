def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if a[i] == a[k] and a[i] != a[j]:
                    count += 1
    print(count)

solve()