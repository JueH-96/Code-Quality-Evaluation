n = int(input())
toys = list(map(int, input().split()))
boxes = list(map(int, input().split()))

def can_assign_with_x(x):
    all_boxes = sorted(boxes + [x])
    toys_sorted = sorted(toys)
    
    # Use greedy assignment: for each toy, assign to smallest fitting box
    used = [False] * len(all_boxes)
    
    for toy in toys_sorted:
        assigned = False
        for i in range(len(all_boxes)):
            if not used[i] and all_boxes[i] >= toy:
                used[i] = True
                assigned = True
                break
        if not assigned:
            return False
    return True

# Binary search on x
left, right = 1, 10**9
result = -1

while left <= right:
    mid = (left + right) // 2
    if can_assign_with_x(mid):
        result = mid
        right = mid - 1
    else:
        left = mid + 1

print(result)