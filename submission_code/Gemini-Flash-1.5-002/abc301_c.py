# YOUR CODE HERE
s = input()
t = input()
s_list = list(s)
t_list = list(t)
s_count = {}
t_count = {}
for char in s_list:
    if char in s_count:
        s_count[char] += 1
    else:
        s_count[char] = 1
for char in t_list:
    if char in t_count:
        t_count[char] += 1
    else:
        t_count[char] = 1
s_at = s_count.get('@', 0)
t_at = t_count.get('@', 0)
s_other = {}
t_other = {}
for char in s_count:
    if char != '@':
        s_other[char] = s_count[char]
for char in t_count:
    if char != '@':
        t_other[char] = t_count[char]
diff = {}
for char in s_other:
    if char in t_other:
        diff[char] = s_other[char] - t_other[char]
    else:
        diff[char] = s_other[char]
for char in t_other:
    if char not in s_other:
        diff[char] = -t_other[char]
at_needed = 0
for char in diff:
    if diff[char] > 0:
        at_needed += diff[char]
if at_needed > s_at + t_at:
    print('No')
else:
    print('Yes')