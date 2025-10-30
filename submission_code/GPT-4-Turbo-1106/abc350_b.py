# Read the number of teeth (N) and the number of treatments (Q)
N, Q = map(int, input().split())

# Initialize a set to keep track of the holes with teeth
teeth = set(range(1, N + 1))

# Read the treatments and perform them
for _ in range(Q):
    T_i = int(input())
    if T_i in teeth:
        teeth.remove(T_i)
    else:
        teeth.add(T_i)

# Output the number of teeth after all treatments
print(len(teeth))