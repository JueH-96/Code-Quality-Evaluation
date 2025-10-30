def can_find_subsequence(s, target):
    i = j = 0
    while i < len(s) and j < len(target):
        if s[i].upper() == target[j]:
            j += 1
        i += 1
    return j == len(target)

s = input().strip()
t = input().strip()

if can_find_subsequence(s, t):
    print("Yes")
elif t.endswith('X') and can_find_subsequence(s, t[:-1]):
    print("Yes")
else:
    print("No")