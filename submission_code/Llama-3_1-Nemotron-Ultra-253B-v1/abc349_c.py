def is_subsequence(t, s):
    t_iter = iter(s)
    for c in t:
        found = False
        for s_char in t_iter:
            if s_char.upper() == c:
                found = True
                break
        if not found:
            return False
    return True

S = input().strip()
T = input().strip()

case1 = is_subsequence(T, S)
case2 = T[2] == 'X' and is_subsequence(T[:2], S)

print("Yes" if case1 or case2 else "No")