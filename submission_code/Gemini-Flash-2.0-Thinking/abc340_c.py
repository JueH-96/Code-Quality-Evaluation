def solve():
    n = int(input())
    total_cost = 0
    queue = [n]
    while queue:
        current = queue.pop(0)
        if current >= 2:
            total_cost += current
            queue.append(current // 2)
            queue.append((current + 1) // 2)
    print(total_cost)

solve()