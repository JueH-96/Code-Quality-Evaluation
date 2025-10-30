n, m = map(int, input().split())
s = input().strip()
t = input().strip()

# Check if all characters in s are present in t
t_chars = set(t)
for c in s:
    if c not in t_chars:
        print("No")
        exit()

# For each position in s, check if there's a valid start position
possible = True
for i in range(n):
    found = False
    # The valid start positions for i are from max(0, i - m + 1) to min(i, n - m)
    start_min = max(0, i - m + 1)
    start_max = min(i, n - m)
    for s_start in range(start_min, start_max + 1):
        # Check if T's character at (i - s_start) matches S[i]
        if s_start + m > n:
            continue
        if t[i - s_start] == s[i]:
            found = True
            break
    if not found:
        possible = False
        break

print("Yes" if possible else "No")