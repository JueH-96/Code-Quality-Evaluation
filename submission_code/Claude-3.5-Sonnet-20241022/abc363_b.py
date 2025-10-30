N, T, P = map(int, input().split())
L = list(map(int, input().split()))

def check(day):
    count = 0
    for length in L:
        if length + day >= T:
            count += 1
    return count >= P

# Check if condition is already satisfied
if check(0):
    print(0)
    exit()

# Binary search for first day when condition is satisfied
left = 1
right = T + 1

while left < right:
    mid = (left + right) // 2
    if check(mid):
        right = mid
    else:
        left = mid + 1

print(left)