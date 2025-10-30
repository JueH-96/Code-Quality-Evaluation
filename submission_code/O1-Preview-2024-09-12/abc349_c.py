# YOUR CODE HERE

def is_subsequence(s, t):
    s_idx = 0
    s_len = len(s)
    for c in t:
        while s_idx < s_len and s[s_idx] != c:
            s_idx +=1
        if s_idx == s_len:
            return False
        s_idx +=1
    return True

S = input().strip()
T = input().strip()

if is_subsequence(S, T.lower()):
    print("Yes")
elif T[2]=='X' and is_subsequence(S, T[:2].lower()):
    print("Yes")
else:
    print("No")