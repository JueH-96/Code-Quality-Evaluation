S, T = input().split()

def check_substring(S, T, w):
    for c in range(1, w+1):
        if S[c-1:w] == T[:c]:
            return True
    return False

w = len(S)
for c in range(1, w+1):
    if check_substring(S, T, c):
        print('Yes')
        break
else:
    print('No')