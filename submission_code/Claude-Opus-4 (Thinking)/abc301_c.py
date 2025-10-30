# YOUR CODE HERE
from collections import Counter

s = input().strip()
t = input().strip()

allowed = set('atcoder')

freq_s = Counter(s)
freq_t = Counter(t)

# Check non-allowed characters (excluding @)
all_chars = set(freq_s.keys()) | set(freq_t.keys())
for c in all_chars:
    if c not in allowed and c != '@':
        if freq_s.get(c, 0) != freq_t.get(c, 0):
            print("No")
            exit()

# Calculate diff for allowed characters
diff = {}
for c in allowed:
    diff[c] = freq_t.get(c, 0) - freq_s.get(c, 0)

# Check if sum of diffs matches @ count difference
num_at_s = freq_s.get('@', 0)
num_at_t = freq_t.get('@', 0)
if num_at_s - num_at_t != sum(diff.values()):
    print("No")
    exit()

# Check if we can satisfy the negative diffs with @ in T
sum_neg_diff = sum(max(0, -d) for d in diff.values())
if sum_neg_diff > num_at_t:
    print("No")
    exit()

# Check if we can satisfy the positive diffs with @ in S
sum_pos_diff = sum(max(0, d) for d in diff.values())
if sum_pos_diff > num_at_s:
    print("No")
    exit()

print("Yes")