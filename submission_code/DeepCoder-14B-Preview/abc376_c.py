n = int(input())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

A_sorted = sorted(A)
B_sorted = sorted(B)

low = 1
high = 10**18
ans = -1

while low <= high:
    mid = (low + high) // 2
    new_boxes = B_sorted + [mid]
    new_boxes_sorted = sorted(new_boxes)
    valid = True
    for i in range(n):
        if new_boxes_sorted[i] < A_sorted[i]:
            valid = False
            break
    if valid:
        ans = mid
        high = mid - 1
    else:
        low = mid + 1

print(ans if ans != -1 else -1)