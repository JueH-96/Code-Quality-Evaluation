import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
Q = int(data[1])
S = data[2]
queries = []

index = 3
for _ in range(Q):
    X_i = int(data[index]) - 1  # Convert to 0-based index
    C_i = data[index + 1]
    queries.append((X_i, C_i))
    index += 2

# Initialize a list to keep track of the counts of "ABC"
counts = [0] * (N - 2)

# Initial count of "ABC" in the string
for i in range(N - 2):
    if S[i:i+3] == "ABC":
        counts[i] = 1

total_count = sum(counts)

results = []

for X_i, C_i in queries:
    # Update the string
    original_char = S[X_i]
    S = S[:X_i] + C_i + S[X_i+1:]

    # Update the counts
    if X_i >= 1 and X_i <= N - 3:
        if S[X_i-1:X_i+2] == "ABC":
            counts[X_i-1] = 1
        else:
            counts[X_i-1] = 0

    if X_i >= 2 and X_i <= N - 2:
        if S[X_i-2:X_i+1] == "ABC":
            counts[X_i-2] = 1
        else:
            counts[X_i-2] = 0

    if X_i >= 0 and X_i <= N - 4:
        if S[X_i:X_i+3] == "ABC":
            counts[X_i] = 1
        else:
            counts[X_i] = 0

    total_count = sum(counts)
    results.append(total_count)

sys.stdout.write("
".join(map(str, results)) + "
")