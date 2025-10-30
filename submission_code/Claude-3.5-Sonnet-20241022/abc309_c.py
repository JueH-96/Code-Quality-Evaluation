N, K = map(int, input().split())
medicines = []
for _ in range(N):
    a, b = map(int, input().split())
    medicines.append((a, b))

def pills_on_day(day):
    total = 0
    for duration, pills in medicines:
        if day <= duration:
            total += pills
    return total

# Binary search for first day with K or fewer pills
left = 1
right = max(a for a,b in medicines) + 1

while left < right:
    mid = (left + right) // 2
    if pills_on_day(mid) <= K:
        right = mid
    else:
        left = mid + 1

print(left)