def solve():
    n, m = map(int, input().split())
    boxes = []
    for _ in range(n):
        boxes.append(list(map(int, input().split())))

    boxes.sort(key=lambda x: x[1] / x[0])

    total_profit = 0
    for v, p in boxes:
        if p < v:
            total_profit += v - p

    print(total_profit)

t = int(input())
for _ in range(t):
    solve()