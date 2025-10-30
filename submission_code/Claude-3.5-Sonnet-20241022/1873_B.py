def solve():
    t = int(input())
    for _ in range(t):
        n = int(input())
        a = list(map(int, input().split()))
        
        # Try adding 1 to each digit and find max product
        max_prod = 0
        for i in range(n):
            # Calculate product with current digit increased by 1
            prod = 1
            for j in range(n):
                if j == i:
                    prod *= (a[j] + 1)
                else:
                    prod *= a[j]
            max_prod = max(max_prod, prod)
        
        print(max_prod)

solve()