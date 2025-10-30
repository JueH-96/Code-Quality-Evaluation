def solve(n, m, boxes):
    if n == 0:
        return 0

    boxes.sort(key=lambda x: x[1])
    prices = [box[1] for box in boxes]
    ans = 0
    min_price = 0
    added = set()
    type_count = 0
    current_box = -1

    def optimize_boxes(boxes):
        nonlocal min_price, added, type_count, current_box
        added = set()
        
        while len(boxes) > type_count:
            if current_box == -1:
                min_price += boxes[0][1]
                current_box = 0
            box = boxes[current_box]
            if box[0] not in added and box[1] > min_price:
                min_price -= prices[current_box]
                added.add(box[0])
                type_count += 1
            current_box += 1
        return min_price + len(added) - type_count

    ans = optimize_boxes(boxes)
    while True:
        if len(added) == m:
            break
        min_price = optimize_boxes(boxes)
        ans += min_price + len(added) - type_count
        type_count = len(added)

    return ans * -1

T = int(input())
for _ in range(T):
    N, M = map(int, input().split())
    boxes = [tuple(map(int, input().split())) for _ in range(N)]
    print(solve(N, M, boxes))