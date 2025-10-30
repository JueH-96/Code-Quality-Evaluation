n, m = map(int, input().split())
x = list(map(int, input().split()))
a = list(map(int, input().split()))

total_stones = sum(a)
if total_stones != n:
    print(-1)
else:
    piles = sorted(zip(x, a))
    current_max = 0
    possible = True
    for xi, ai in piles:
        if xi > current_max + 1:
            possible = False
            break
        end = xi + ai - 1
        if end > n:
            end = n
        if end > current_max:
            current_max = end
    if current_max < n:
        possible = False
    
    if not possible:
        print(-1)
    else:
        total_ops = 0
        for xi, ai in piles:
            total_ops += ai * (ai - 1) // 2
        print(total_ops)