n = int(input())
s = list(map(int, input().split()))

# Convert to set for O(1) lookup
s_set = set(s)

count = 0
# For each element as the middle element b
for b in s:
    # For each element a < b, check if c = 2b - a is in the set
    for a in s:
        if a < b:
            c = 2 * b - a
            if c in s_set:
                count += 1

print(count)