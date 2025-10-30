def solve():
    n = int(input())
    a = list(map(int, input().split()))
    w = list(map(int, input().split()))

    counts = {}
    for x in a:
        counts[x] = counts.get(x, 0) + 1

    items_in_box = {}
    for i in range(n):
        if a[i] not in items_in_box:
            items_in_box[a[i]] = []
        items_in_box[a[i]].append((w[i], i + 1))

    empty_boxes = []
    for i in range(1, n + 1):
        if i not in counts:
            empty_boxes.append(i)

    overloaded_boxes = []
    for box, count in counts.items():
        if count > 1:
            overloaded_boxes.append(box)

    total_cost = 0
    
    for box in overloaded_boxes:
        items_in_box[box].sort()
        
        for i in range(len(items_in_box[box]) - 1):
            total_cost += items_in_box[box][i][0]

    print(total_cost)

solve()