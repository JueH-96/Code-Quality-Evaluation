# Read input values A and B
A, B = map(int, input().split())

# Create a set of all possible culprits (1, 2, 3)
# Then remove the people that witnesses say are not the culprit
potential_culprits = set(range(1, 4)) - {A, B}

# If there's exactly one potential culprit left, that's our answer
if len(potential_culprits) == 1:
    print(list(potential_culprits)[0])
else:
    # Otherwise, we can't uniquely identify the culprit
    print(-1)