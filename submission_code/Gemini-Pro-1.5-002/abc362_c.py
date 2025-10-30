# YOUR CODE HERE
def solve():
    n = int(input())
    ranges = []
    for _ in range(n):
        l, r = map(int, input().split())
        ranges.append((l, r))

    x = [0] * n
    total_sum = 0
    min_sum = 0
    max_sum = 0

    for i in range(n):
        l, r = ranges[i]
        x[i] = l
        total_sum += l
        min_sum += l
        max_sum += r

    if not (min_sum <= 0 <= max_sum):
        print("No")
        return

    diff = -total_sum
    for i in range(n):
        l, r = ranges[i]
        add = min(diff, r - x[i])
        x[i] += add
        diff -= add
        if diff == 0:
            break
    
    print("Yes")
    print(*x)


solve()