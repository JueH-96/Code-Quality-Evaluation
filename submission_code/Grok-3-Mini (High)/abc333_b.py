import sys

# Read input from stdin
data = sys.stdin.read().split()
s_pair = data[0]
t_pair = data[1]

# Convert characters to indices (A=0, B=1, C=2, D=3, E=4)
idx_s1 = ord(s_pair[0]) - ord('A')
idx_s2 = ord(s_pair[1]) - ord('A')
idx_t1 = ord(t_pair[0]) - ord('A')
idx_t2 = ord(t_pair[1]) - ord('A')

# Calculate the minimum graph distance for each pair
dist_s = min(abs(idx_s1 - idx_s2), 5 - abs(idx_s1 - idx_s2))
dist_t = min(abs(idx_t1 - idx_t2), 5 - abs(idx_t1 - idx_t2))

# Compare the distances and output the result
if dist_s == dist_t:
    print("Yes")
else:
    print("No")