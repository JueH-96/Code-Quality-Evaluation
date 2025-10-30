# YOUR CODE HERE
N, M = map(int, input().split())
stands = []
for _ in range(N):
    s = input().strip()
    stands.append(s)

# Convert each stand to a bitmask representing flavors it sells
stand_masks = []
for stand in stands:
    mask = 0
    for j in range(M):
        if stand[j] == 'o':
            mask |= (1 << j)
    stand_masks.append(mask)

# All flavors mask (all M bits set)
all_flavors = (1 << M) - 1

min_stands = N  # worst case is visiting all stands

# Try all possible combinations of stands
for combination in range(1, 1 << N):
    # Count number of stands in this combination
    num_stands = bin(combination).count('1')
    
    # Skip if this combination already uses more stands than our current minimum
    if num_stands >= min_stands:
        continue
    
    # Calculate which flavors are covered by this combination
    covered_flavors = 0
    for i in range(N):
        if combination & (1 << i):
            covered_flavors |= stand_masks[i]
    
    # Check if all flavors are covered
    if covered_flavors == all_flavors:
        min_stands = num_stands

print(min_stands)