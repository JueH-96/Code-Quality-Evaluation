def can_match(toys, boxes):
    n = len(toys)
    j = 0
    for i in range(n):
        # Find first box that can fit toys[i]
        while j < n and boxes[j] < toys[i]:
            j += 1
        # Check Hall's condition: need at least (n-i) boxes for remaining toys
        if j > i:
            return False
    return True

n = int(input())
toys = list(map(int, input().split()))
boxes = list(map(int, input().split()))

toys.sort()

# First check if it's possible at all (with very large x)
all_boxes = boxes + [10**10]
all_boxes.sort()
if not can_match(toys, all_boxes):
    print(-1)
else:
    # Binary search for minimum x
    left, right = 1, max(toys)
    result = -1
    
    while left <= right:
        mid = (left + right) // 2
        all_boxes = boxes + [mid]
        all_boxes.sort()
        
        if can_match(toys, all_boxes):
            result = mid
            right = mid - 1
        else:
            left = mid + 1
    
    print(result)