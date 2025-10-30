import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
K = int(data[1])

days = []
for i in range(N):
    a = int(data[2 + 2 * i])
    b = int(data[3 + 2 * i])
    days.append((a, b))

days.sort()

def can_take_k_pills(day, K):
    total = 0
    for a, b in days:
        if day >= a:
            total += b
        if total > K:
            return False
    return True

left, right = 1, 10**9
while left < right:
    mid = (left + right) // 2
    if can_take_k_pills(mid, K):
        right = mid
    else:
        left = mid + 1

print(left)