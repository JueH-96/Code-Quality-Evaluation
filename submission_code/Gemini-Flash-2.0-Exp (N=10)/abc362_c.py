def solve():
    n = int(input())
    ranges = []
    for _ in range(n):
        l, r = map(int, input().split())
        ranges.append((l, r))

    min_sum = 0
    max_sum = 0
    for l, r in ranges:
        min_sum += l
        max_sum += r
    
    if min_sum > 0 or max_sum < 0:
        print("No")
        return
    
    x = []
    current_sum = 0
    for l, r in ranges:
        x_i = max(l, min(r, -current_sum))
        x.append(x_i)
        current_sum += x_i

    if current_sum != 0:
        print("No")
        return
    
    print("Yes")
    print(*x)

solve()