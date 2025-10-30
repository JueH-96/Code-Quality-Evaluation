# Read input
N, K = map(int, input().split())
A = list(map(int, input().split()))

current_seats = K
count = 0
groups = A.copy()

while True:
    if not groups:
        count += 1
        break
    if current_seats < groups[0]:
        count += 1
        current_seats = K
    else:
        current_seats -= groups[0]
        groups.pop(0)

print(count)