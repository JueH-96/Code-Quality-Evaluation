def can_match_subsequence(s, t):
    """Returns the maximum length of a prefix of t that can be matched as a subsequence in s."""
    i, j = 0, 0
    while i < len(s) and j < len(t):
        if s[i] == t[j]:
            j += 1
        i += 1
    return j

n, t = input().split()
n = int(n)
strings = []
for _ in range(n):
    strings.append(input().strip())

count = 0
for i in range(n):
    for j in range(n):
        # Check if strings[i] + strings[j] contains t as a subsequence
        max_prefix = can_match_subsequence(strings[i], t)
        remaining = t[max_prefix:]
        if can_match_subsequence(strings[j], remaining) == len(remaining):
            count += 1

print(count)