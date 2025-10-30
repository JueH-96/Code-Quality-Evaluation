def can_delete(s, t):
    if len(s) != len(t) + 1:
        return False
    i = j = 0
    while i < len(s) and j < len(t):
        if s[i] == t[j]:
            i += 1
            j += 1
        else:
            i += 1
            break
    # Now check if the remaining parts match
    return (s[i:] == t[j:]) or (len(s[i:]) == len(t[j:]) + 1 and s[i+1:] == t[j:] and (len(s[i:]) == len(t[j:]) + 1))

K = int(input())
S = input().strip()
T = input().strip()

if len(S) == len(T):
    diff = 0
    for i in range(len(S)):
        if S[i] != T[i]:
            diff += 1
            if diff > 1:
                break
    if diff == 1:
        print("Yes")
    else:
        print("No")
elif abs(len(S) - len(T)) == 1:
    if len(S) > len(T):
        if can_delete(S, T):
            print("Yes")
        else:
            print("No")
    else:
        if can_delete(T, S):
            print("Yes")
        else:
            print("No")
else:
    print("No")