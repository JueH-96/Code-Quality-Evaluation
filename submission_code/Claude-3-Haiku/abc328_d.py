s = input()

while 'ABC' in s:
    i = s.index('ABC')
    s = s[:i] + s[i+3:]

print(s)