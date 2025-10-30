# YOUR CODE HERE
def can_make_abc(s):
    if s == 'abc':
        return 'YES'
    if s[0] == 'a' or s[1] == 'b' or (s[0] == 'b' and s[2] == 'c'):
        return 'YES'
    return 'NO'

t = int(input())
for _ in range(t):
    s = input().strip()
    print(can_make_abc(s))