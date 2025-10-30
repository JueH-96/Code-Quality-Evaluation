N = int(input())
S = input()
T = input()

NOK = 'No'
OK = 'Yes'

for s, t in zip(S, T):
    if (s, t) in {
        ('l', '1'),
        ('1', 'l'),
        ('0', 'o'),
        ('o', '0'),
    }:
        continue
    elif s != t:
        print(NOK)
        exit()

print(OK)