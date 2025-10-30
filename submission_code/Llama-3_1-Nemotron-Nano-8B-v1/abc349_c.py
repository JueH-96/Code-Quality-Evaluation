s = input().strip()
t = input().strip()

def is_subsequence(target, s):
    ptr = 0
    n = len(s)
    for c in target:
        while ptr < n and s[ptr] != c:
            ptr += 1
        if ptr >= n:
            return False
        ptr += 1
    return True

case1 = is_subsequence(t, s)
case2 = (t[2] == 'X') and is_subsequence(t[:2], s)

print("Yes" if case1 or case2 else "No")