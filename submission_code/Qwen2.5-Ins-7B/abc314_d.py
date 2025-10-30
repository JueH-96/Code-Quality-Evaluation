# YOUR CODE HERE
n = int(input())
s = input()
q = int(input())
for _ in range(q):
    t, x, c = map(int, input().split()), input()
    if t[0] == 1:
        s = s[:t[1]-1] + t[2] + s[t[1]:]
    elif t[0] == 2:
        s = s.lower()
    else:
        s = s.upper()
print(s)