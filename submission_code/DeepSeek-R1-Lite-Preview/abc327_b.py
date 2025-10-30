B = int(input())

low = 1
high = 1

# Expand the upper bound exponentially
while high ** high < B:
    high *= 2

# Binary search to find A such that A^A == B
while low <= high:
    mid = (low + high) // 2
    mid_power = mid ** mid
    if mid_power == B:
        print(mid)
        break
    elif mid_power < B:
        low = mid + 1
    else:
        high = mid - 1
else:
    print(-1)