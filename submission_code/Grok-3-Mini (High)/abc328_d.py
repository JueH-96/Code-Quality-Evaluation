import sys
s = sys.stdin.read().strip()
while 'ABC' in s:
    pos = s.find('ABC')
    s = s[:pos] + s[pos+3:]
print(s)