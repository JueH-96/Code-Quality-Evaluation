def can_place_toys(N, toys, boxes, new_box_size):
    # Create list of all boxes including the new box
    all_boxes = boxes + [new_box_size]
    
    # Sort toys and boxes in descending order
    toys = sorted(toys, reverse=True)
    all_boxes = sorted(all_boxes, reverse=True)
    
    # Try to place each toy in the first box that can fit it
    for i in range(N):
        if i >= N or toys[i] > all_boxes[i]:
            return False
    return True

def solve():
    # Read input
    N = int(input())
    toys = list(map(int, input().split()))
    boxes = list(map(int, input().split()))
    
    # If any toy can't fit in any box (including potential new box),
    # it's impossible
    max_toy = max(toys)
    max_box = max(boxes)
    
    # Binary search for the minimum new box size
    left = 1
    right = 10**9 + 1
    result = -1
    
    while left <= right:
        mid = (left + right) // 2
        if can_place_toys(N, toys, boxes, mid):
            result = mid
            right = mid - 1
        else:
            left = mid + 1
    
    print(result)

# Run the solution
solve()