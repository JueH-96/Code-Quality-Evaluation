def solve():
    n = int(input())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    k = int(input())
    
    for _ in range(k):
        x, y = map(int, input().split())
        total_sum = 0
        for i in range(x):
            for j in range(y):
                total_sum += abs(a[i] - b[j])
        print(total_sum)

solve()