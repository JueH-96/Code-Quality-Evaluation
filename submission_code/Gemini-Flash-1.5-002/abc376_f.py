def solve():
    n, q = map(int, input().split())
    instructions = []
    for _ in range(q):
        instructions.append(input().split())

    def dist(a, b, n):
        d = abs(a - b)
        return min(d, n - d)

    left = 1
    right = 2
    total_ops = 0

    for i in range(q):
        hand, target = instructions[i]
        target = int(target)

        if hand == 'L':
            ops = dist(left, target, n)
            total_ops += ops
            left = target
            
            if dist(right, target, n) <= dist(right, left, n) and dist(right, target, n) != 0:
                
                if right != target:
                    total_ops += dist(right, target, n)
                    right = target
            
        else:
            ops = dist(right, target, n)
            total_ops += ops
            right = target
            
            if dist(left, target, n) <= dist(left, right, n) and dist(left, target, n) != 0:
                if left != target:
                    total_ops += dist(left, target, n)
                    left = target

    print(total_ops)

solve()