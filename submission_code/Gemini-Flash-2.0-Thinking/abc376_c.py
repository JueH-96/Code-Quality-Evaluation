def solve():
    n = int(input())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))

    def can_place(toys, boxes):
        toys = sorted(toys)
        boxes = sorted(boxes)
        box_idx = 0
        for toy in toys:
            while box_idx < len(boxes) and boxes[box_idx] < toy:
                box_idx += 1
            if box_idx == len(boxes):
                return False
            box_idx += 1
        return True

    possible_x = sorted(list(set(a)))

    for x in possible_x:
        boxes_with_x = sorted(b + [x])
        if can_place(a, boxes_with_x):
            print(x)
            return

    print(-1)

if __name__ == "__main__":
    solve()