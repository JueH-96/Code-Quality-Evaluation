# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))
w = list(map(int, input().split()))

box_count = {}
for x in a:
    if x not in box_count:
        box_count[x] = 0
    box_count[x] += 1

item_in_box = {}
for i in range(n):
    if a[i] not in item_in_box:
        item_in_box[a[i]] = []
    item_in_box[a[i]].append((w[i], i))

for box in item_in_box:
    item_in_box[box].sort(reverse=True)

ans = 0
extra_items = []
empty_boxes = []
for i in range(1, n + 1):
    if i in box_count:
        if box_count[i] > 1:
            for j in range(1, box_count[i]):
                extra_items.append(item_in_box[i][j])
        elif box_count[i] == 1:
            pass
    else:
        empty_boxes.append(i)

extra_items.sort(reverse=True)
empty_boxes.sort()

for i in range(len(extra_items)):
    ans += extra_items[i][0]

print(ans)