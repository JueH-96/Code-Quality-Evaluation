# Read input
S = input().strip()
n = len(S)

# Compute prefix counts
prefix = [ [0] * 26 ]
for c in S:
    new_row = prefix[-1].copy()
    idx = ord(c) - ord('A')
    new_row[idx] += 1
    prefix.append(new_row)

# Compute suffix counts
suffix = [ [0] * 26 for _ in range(n) ]
suffix_counts = [0] * 26
for j in range(n-1, -1, -1):
    suffix[j] = suffix_counts.copy()
    c = S[j]
    idx = ord(c) - ord('A')
    suffix_counts[idx] += 1

# Calculate total triples
total = 0
for j in range(n):
    left = prefix[j]
    right = suffix[j]
    total += sum(left[c] * right[c] for c in range(26))

print(total)