def can_match(toys, boxes):
    # Sort both arrays
    toys_sorted = sorted(toys)
    boxes_sorted = sorted(boxes)
    
    n = len(toys)
    
    # Try to match each toy to a box
    box_idx = 0
    for toy in toys_sorted:
        # Find the first box that can hold this toy
        while box_idx < n and boxes_sorted[box_idx] < toy:
            box_idx += 1
        
        if box_idx >= n:
            # No box available for this toy
            return False
        
        # Assign this toy to the box
        box_idx += 1
    
    return True

def solve():
    n = int(input())
    toys = list(map(int, input().split()))
    existing_boxes = list(map(int, input().split()))
    
    # Binary search on the size of the new box
    left, right = 1, max(toys)
    result = -1
    
    while left <= right:
        mid = (left + right) // 2
        
        # Check if we can match all toys with existing boxes + new box of size mid
        all_boxes = existing_boxes + [mid]
        
        if can_match(toys, all_boxes):
            result = mid
            right = mid - 1
        else:
            left = mid + 1
    
    print(result)

solve()