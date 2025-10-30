n = int(input())
perm = list(map(int, input().split()))

arr = perm[:]
total_cost = 0

# Bubble sort with cost tracking
while True:
    swapped = False
    for i in range(n-1):
        if arr[i] > arr[i+1]:
            arr[i], arr[i+1] = arr[i+1], arr[i]
            total_cost += (i + 1)  # Cost is i+1 for swapping positions i and i+1 (1-indexed)
            swapped = True
    if not swapped:
        break

print(total_cost)