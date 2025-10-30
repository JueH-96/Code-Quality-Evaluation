def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    count = 0
    
    for i in range(n):
        for j in range(n):
            if i != j:
                if a[i] <= a[j] / 2:
                    count += 1
    
    print(count)

solve()