def solve():
    n = int(input())
    products = []
    for _ in range(n):
        t, d = map(int, input().split())
        products.append((t, t + d))

    products.sort(key=lambda x: x[1])
    
    count = 0
    last_print_time = -float('inf')
    
    for start, end in products:
        if start >= last_print_time + 1:
            count += 1
            last_print_time = end
        
    print(count)

solve()