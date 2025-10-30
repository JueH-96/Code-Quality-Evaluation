def is_similar(s, t):
    for i in range(len(s)):
        if s[i] != t[i]:
            if (s[i] not in ['1', 'l'] or t[i] not in ['1', 'l']) and (s[i] not in ['0', 'o'] or t[i] not in ['0', 'o']):
                return False
    return True

N = int(input())
S = input()
T = input()

if is_similar(S, T):
    print('Yes')
else:
    print('No')