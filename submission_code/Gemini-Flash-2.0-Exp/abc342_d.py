def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            product = a[i] * a[j]
            
            if product == 0:
                count += 1
                continue
            
            sqrt_product = int(product**0.5)
            if sqrt_product * sqrt_product == product:
                count += 1
    
    print(count)

solve()