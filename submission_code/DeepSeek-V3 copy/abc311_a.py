# Read input
N = int(input())
S = input().strip()

# Initialize a set to keep track of seen characters
seen = set()
# Iterate through the string
for i in range(N):
    seen.add(S[i])
    # Check if all three characters are in the set
    if len(seen) == 3:
        print(i + 1)
        break