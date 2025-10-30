import sys
from sys import stdin, stdout

def flip(s, l, r):
    s = list(s)
    for i in range(l, r+1):
        if s[i] == '0':
            s[i] = '1'
        else:
            s[i] = '0'
    return ''.join(s)

def is_good(s, l, r):
    for i in range(l, r):
        if s[i] == s[i+1]:
            return False
    return True

def main():
    n, q = map(int, stdin.readline().strip().split())
    s = stdin.readline().strip()
    for _ in range(q):
        query = list(map(int, stdin.readline().strip().split()))
        if query[0] == 1:
            s = flip(s, query[1]-1, query[2]-1)
        else:
            if is_good(s, query[1]-1, query[2]-1):
                stdout.write('Yes
')
            else:
                stdout.write('No
')

if __name__ == "__main__":
    main()