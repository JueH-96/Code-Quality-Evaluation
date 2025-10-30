# YOUR CODE HERE
n, m = map(int, input().split())
a = list(map(int, input().split()))
b = list(map(int, input().split()))

# Sort both arrays
a.sort()
b.sort()

# Try to match each person with a box using two pointers
total_cost = 0
j = 0  # pointer for boxes

for i in range(m):
    # Find the cheapest box that satisfies person i's requirement
    while j < n and a[j] < b[i]:
        j += 1
    
    if j >= n:
        # No suitable box found
        print(-1)
        exit()
    
    # Assign this box to person i
    total_cost += a[j]
    j += 1

print(total_cost)