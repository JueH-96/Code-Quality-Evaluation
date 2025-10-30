def longest_prefix_match(s, t):
    """Returns the length of the longest prefix of t that can be matched as a subsequence in s"""
    if not t:
        return 0
    
    i = 0  # pointer for s
    j = 0  # pointer for t
    
    while i < len(s) and j < len(t):
        if s[i] == t[j]:
            j += 1
        i += 1
    
    return j

def longest_suffix_match(s, t):
    """Returns the length of the longest suffix of t that can be matched as a subsequence in s"""
    if not t:
        return 0
    
    i = len(s) - 1  # pointer for s (from end)
    j = len(t) - 1  # pointer for t (from end)
    
    while i >= 0 and j >= 0:
        if s[i] == t[j]:
            j -= 1
        i -= 1
    
    return len(t) - 1 - j

# Read input
line = input().split()
N = int(line[0])
T = line[1]

strings = []
for _ in range(N):
    strings.append(input().strip())

# Precompute prefix and suffix matches
prefix_matches = []
suffix_matches = []

for s in strings:
    prefix_matches.append(longest_prefix_match(s, T))
    suffix_matches.append(longest_suffix_match(s, T))

# Count valid pairs
count = 0
for i in range(N):
    for j in range(N):
        if prefix_matches[i] + suffix_matches[j] >= len(T):
            count += 1

print(count)