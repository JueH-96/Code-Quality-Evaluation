def solve():
    n, q = map(int, input().split())
    r = list(map(int, input().split()))
    
    sorted_r = sorted(enumerate(r, 1), key=lambda x: x[1])
    
    for _ in range(q):
        x = int(input())
        
        count = 0
        current_sum = 0
        for _, req in sorted_r:
            if current_sum + req <= x:
                current_sum += req
                count += 1
            else:
                break
        print(count)

solve()