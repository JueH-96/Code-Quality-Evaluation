from collections import Counter

s = input().strip()
t = input().strip()

allowed = {'a', 't', 'c', 'o', 'd', 'e', 'r'}

# Check non-allowed characters (excluding '@')
non_allowed_s = Counter()
for c in s:
    if c != '@' and c not in allowed:
        non_allowed_s[c] += 1

non_allowed_t = Counter()
for c in t:
    if c != '@' and c not in allowed:
        non_allowed_t[c] += 1

# Verify counts for non-allowed characters
all_non_allowed = set(non_allowed_s.keys()) | set(non_allowed_t.keys())
for c in all_non_allowed:
    if non_allowed_s.get(c, 0) != non_allowed_t.get(c, 0):
        print("No")
        exit()

# Process allowed characters and '@' counts
s_counts = Counter()
at_s = 0
for c in s:
    if c == '@':
        at_s += 1
    elif c in allowed:
        s_counts[c] += 1

t_counts = Counter()
at_t = 0
for c in t:
    if c == '@':
        at_t += 1
    elif c in allowed:
        t_counts[c] += 1

# Calculate sum of minimum required y_c
sum_min_y = 0
for c in allowed:
    sc = s_counts.get(c, 0)
    tc = t_counts.get(c, 0)
    sum_min_y += max(0, sc - tc)

if sum_min_y <= at_t:
    print("Yes")
else:
    print("No")