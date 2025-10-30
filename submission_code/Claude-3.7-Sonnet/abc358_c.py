from itertools import combinations

# Read input
N, M = map(int, input().split())
stands = [input() for _ in range(N)]

# Convert each stand's offerings to a set of flavors
flavor_sets = []
for stand in stands:
    flavors = set()
    for i, char in enumerate(stand):
        if char == 'o':
            flavors.add(i + 1)
    flavor_sets.append(flavors)

# All flavors that Takahashi wants to try
all_flavors = set(range(1, M + 1))

# Try combinations starting from the smallest number of stands
for num_stands in range(1, N + 1):
    for combo in combinations(range(N), num_stands):
        # Calculate the union of flavors for this combination of stands
        available_flavors = set()
        for stand_idx in combo:
            available_flavors.update(flavor_sets[stand_idx])
        
        # If this combination covers all flavors, return the number of stands
        if available_flavors == all_flavors:
            print(num_stands)
            exit()