# YOUR CODE HERE
def can_place_toys(toys, boxes, x):
    boxes = sorted(boxes + [x], reverse=True)
    toys = sorted(toys, reverse=True)
    
    for toy, box in zip(toys, boxes):
        if toy > box:
            return False
    return True

def find_minimum_x(N, toys, boxes):
    left, right = 1, max(toys)
    
    while left <= right:
        mid = (left + right) // 2
        if can_place_toys(toys, boxes, mid):
            right = mid - 1
        else:
            left = mid + 1
    
    return left if can_place_toys(toys, boxes, left) else -1

# Read input
N = int(input())
toys = list(map(int, input().split()))
boxes = list(map(int, input().split()))

# Find and print the result
result = find_minimum_x(N, toys, boxes)
print(result)