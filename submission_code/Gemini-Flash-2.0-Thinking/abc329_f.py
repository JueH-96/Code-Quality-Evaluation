def solve():
    n, q = map(int, input().split())
    colors = list(map(int, input().split()))
    boxes = [set([colors[i]]) for i in range(n)]

    for _ in range(q):
        a, b = map(int, input().split())
        a -= 1
        b -= 1

        balls_a = boxes[a]
        boxes[b].update(balls_a)
        boxes[a] = set()
        print(len(boxes[b]))

solve()