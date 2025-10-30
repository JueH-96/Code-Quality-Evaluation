import sys
input = sys.stdin.readline

n = int(input())
s = list(input().strip())
q = int(input())

caps_on = 0 

for _ in range(q):
    t, x, c = input().split()
    t, x = int(t), int(x)
    if t == 1:
        s[x - 1] = c
    elif t == 2:
        caps_on ^= 1
    elif t == 3:
        caps_on ^= 1

res = ""
for ch in s:
    if caps_on and ch.islower() or not caps_on and ch.isupper():
        res += ch.upper()
    else:
        res += ch.lower()

print(res)