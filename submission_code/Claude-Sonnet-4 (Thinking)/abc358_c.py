n, m = map(int, input().split())

# Read the stands and convert to sets of flavors
stands = []
for i in range(n):
    s = input()
    flavors = set()
    for j in range(m):
        if s[j] == 'o':
            flavors.add(j)
    stands.append(flavors)

# Try all possible combinations of stands
min_stands = float('inf')

# Use bit manipulation to iterate through all subsets
for mask in range(1, 1 << n):
    covered_flavors = set()
    stand_count = 0
    
    for i in range(n):
        if mask & (1 << i):
            stand_count += 1
            covered_flavors.update(stands[i])
    
    # If all flavors are covered, update minimum
    if len(covered_flavors) == m:
        min_stands = min(min_stands, stand_count)

print(min_stands)