# Read N and Q from input
N, Q = map(int, input().split())

# Read the list of treatments
T = list(map(int, input().split()))

# Initialize the list of teeth, all starting with True (tooth present)
teeth = [True] * N

# Perform each treatment
for t in T:
    # Toggle the tooth in the corresponding hole (adjusting for 0-indexing)
    teeth[t - 1] = not teeth[t - 1]

# Count the number of teeth remaining
count = sum(teeth)

# Print the final count
print(count)