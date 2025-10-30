# YOUR CODE HERE
s = input().strip()
t = input().strip()

s_dist = abs(ord(s[0]) - ord(s[1]))
t_dist = abs(ord(t[0]) - ord(t[1]))

if s_dist == 2 or s_dist == 3:
    s_dist = 5 - s_dist
if t_dist == 2 or t_dist == 3:
    t_dist = 5 - t_dist

if s_dist == t_dist:
    print("Yes")
else:
    print("No")