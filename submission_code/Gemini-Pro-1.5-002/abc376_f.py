def solve():
    n, q = map(int, input().split())
    queries = []
    for _ in range(q):
        h, t = input().split()
        queries.append((h, int(t)))

    left = 1
    right = 2
    total_ops = 0

    for h, t in queries:
        if h == 'L':
            target = t
            current = left
            
            diff1 = abs(target - current)
            diff2 = n - diff1
            
            ops = min(diff1, diff2)
            
            total_ops += ops
            left = target
        else:
            target = t
            current = right
            
            diff1 = abs(target - current)
            diff2 = n - diff1
            
            ops = min(diff1, diff2)
            total_ops += ops
            right = target

    print(total_ops)

solve()