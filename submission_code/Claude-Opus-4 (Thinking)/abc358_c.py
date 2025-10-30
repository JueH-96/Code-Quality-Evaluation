# YOUR CODE HERE
n, m = map(int, input().split())
stands = []
for i in range(n):
    s = input().strip()
    mask = 0
    for j in range(m):
        if s[j] == 'o':
            mask |= (1 << j)
    stands.append(mask)

# Target mask with all flavors
target = (1 << m) - 1

# Try all subsets of stands
min_stands = n + 1
for subset in range(1, 1 << n):
    # Calculate which flavors this subset covers
    covered = 0
    count = 0
    for i in range(n):
        if subset & (1 << i):
            covered |= stands[i]
            count += 1
    
    if covered == target:
        min_stands = min(min_stands, count)

print(min_stands)