import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])

# Binary search to find the minimum number of friends needed
left, right = 1, N
while left < right:
    mid = (left + right) // 2
    print(mid)
    sys.stdout.flush()
    S = input().strip()
    if S.count('1') == 1:
        right = mid
    else:
        left = mid + 1

M = left

# Determine which friend had the stomach upset
for i in range(M):
    if S[i] == '1':
        spoiled_bottle = i + 1
        break

print(spoiled_bottle)
sys.stdout.flush()