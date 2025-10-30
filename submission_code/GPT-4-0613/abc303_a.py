N = int(input().strip())
S = input().strip()
T = input().strip()

def are_similar(S, T):
    for s, t in zip(S, T):
        if s == t:
            continue
        elif (s == '1' and t == 'l') or (s == 'l' and t == '1'):
            continue
        elif (s == '0' and t == 'o') or (s == 'o' and t == '0'):
            continue
        else:
            return False
    return True

if are_similar(S, T):
    print('Yes')
else:
    print('No')