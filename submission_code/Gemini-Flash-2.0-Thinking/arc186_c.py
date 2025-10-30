def solve():
    n, m = map(int, input().split())
    boxes = []
    for _ in range(n):
        v, p = map(int, input().split())
        boxes.append((v, p))

    profit = 0
    for v, p in boxes:
        if v > p:
            profit += (v - p)
    print(profit)

t = int(input())
for _ in range(t):
    solve()