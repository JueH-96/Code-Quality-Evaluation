def solve():
    n = int(input())
    products = []
    for _ in range(n):
        t, d = map(int, input().split())
        products.append((t, t + d))
    
    products.sort(key=lambda x: x[1])
    
    count = 0
    last_printed_time = -1
    
    for start_time, end_time in products:
        if start_time > last_printed_time:
            count += 1
            last_printed_time = end_time - 1
            
    print(count)

solve()