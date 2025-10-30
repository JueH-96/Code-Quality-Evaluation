N, M = map(int, input().split())
stands = [input().strip() for _ in range(N)]

# Convert strings to sets of flavors
flavors = []
for s in stands:
    stand_flavors = {i+1 for i, c in enumerate(s) if c == 'o'}
    flavors.append(stand_flavors)

# All flavors set
all_flavors = set(range(1, M+1))

# Initialize minimum number of stands to a high value
min_stands = N + 1

# Iterate over all possible subsets of stands
for i in range(1, 1 << N):
    union = set()
    count = 0
    for j in range(N):
        if (i >> j) & 1:
            union.update(flavors[j])
            count += 1
    if union == all_flavors:
        if count < min_stands:
            min_stands = count

print(min_stands)