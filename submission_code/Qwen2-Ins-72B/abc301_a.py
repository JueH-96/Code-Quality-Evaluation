N = int(input())
S = input()

t = 0
a = 0
for i in range(N):
    if S[i] == 'T':
        t += 1
    else:
        a += 1
    if t > a and (N - i - 1) < a:
        print('T')
        break
    elif a > t and (N - i - 1) < t:
        print('A')
        break
else:
    if t > a:
        print('T')
    elif a > t:
        print('A')
    else:
        print('T')