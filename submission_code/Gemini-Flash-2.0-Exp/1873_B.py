def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    max_product = 0
    
    for i in range(n):
        original_value = a[i]
        a[i] += 1
        
        product = 1
        for val in a:
            product *= val
        
        max_product = max(max_product, product)
        
        a[i] = original_value
        
    print(max_product)

t = int(input())
for _ in range(t):
    solve()