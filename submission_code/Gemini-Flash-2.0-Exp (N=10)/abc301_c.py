from collections import Counter

s = input()
t = input()

s_count = Counter(s)
t_count = Counter(t)

atcoder = "atcoder"

for char in "abcdefghijklmnopqrstuvwxyz":
    if s_count[char] != t_count[char]:
        if char not in atcoder:
            print("No")
            exit()
        else:
            diff = s_count[char] - t_count[char]
            if diff > 0:
                if s_count['@'] < diff:
                    print("No")
                    exit()
                else:
                    s_count['@'] -= diff
            elif diff < 0:
                if t_count['@'] < -diff:
                    print("No")
                    exit()
                else:
                    t_count['@'] += diff

if s_count['@'] != t_count['@']:
    print("No")
else:
    print("Yes")