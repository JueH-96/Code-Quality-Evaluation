W, B = map(int, input().split())
base = 'wbwbwwbwbwbw'
s = base * 20  # Generate a sufficiently long string

# Compute prefix sums for 'w' and 'b'
prefix_w = [0] * (len(s) + 1)
prefix_b = [0] * (len(s) + 1)
for i in range(len(s)):
    prefix_w[i+1] = prefix_w[i] + (1 if s[i] == 'w' else 0)
    prefix_b[i+1] = prefix_b[i] + (1 if s[i] == 'b' else 0)

found = False
# Check all possible substrings
for i in range(len(s)):
    for j in range(i, len(s)):
        w = prefix_w[j+1] - prefix_w[i]
        b = prefix_b[j+1] - prefix_b[i]
        if w == W and b == B:
            found = True
            break
    if found:
        break

print("Yes" if found else "No")