# YOUR CODE HERE
def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    max_prod = -1
    
    for i in range(n):
        b = list(a)
        b[i] += 1
        prod = 1
        for x in b:
            prod *= x
        max_prod = max(max_prod, prod)
    
    print(max_prod)

t = int(input())
for _ in range(t):
    solve()