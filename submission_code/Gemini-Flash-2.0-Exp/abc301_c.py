S = input()
T = input()

s_counts = {}
t_counts = {}

for char in "atcoder":
    s_counts[char] = 0
    t_counts[char] = 0

s_at = 0
t_at = 0

for char in S:
    if char == "@":
        s_at += 1
    else:
        if char in s_counts:
            s_counts[char] += 1
        else:
            print("No")
            exit()

for char in T:
    if char == "@":
        t_at += 1
    else:
        if char in t_counts:
            t_counts[char] += 1
        else:
            print("No")
            exit()

possible = True
for char in "atcoder":
    diff = s_counts[char] - t_counts[char]
    if diff > 0:
        t_at -= diff
    elif diff < 0:
        s_at += diff
    
    if t_at < 0 or s_at < 0:
        possible = False
        break

if s_at != -t_at and s_at + t_at != 0:
    possible = False

if possible:
    print("Yes")
else:
    print("No")