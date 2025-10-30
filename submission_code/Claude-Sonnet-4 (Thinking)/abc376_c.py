N = int(input())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

def can_assign(x):
    toys = sorted(A)
    boxes = sorted(B + [x])
    
    i, j = 0, 0
    while i < len(toys) and j < len(boxes):
        if toys[i] <= boxes[j]:
            i += 1
            j += 1
        else:
            j += 1
    
    return i == len(toys)

# Binary search on x
left, right = 1, max(A)
result = -1

while left <= right:
    mid = (left + right) // 2
    if can_assign(mid):
        result = mid
        right = mid - 1
    else:
        left = mid + 1

print(result)