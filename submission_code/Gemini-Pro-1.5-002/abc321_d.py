# YOUR CODE HERE
def solve():
    n, m, p = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    
    total_price = 0
    for i in range(n):
        for j in range(m):
            total_price += min(a[i] + b[j], p)
    
    print(total_price)

solve()