# YOUR CODE HERE

import sys

def check_insert(s, t):
    for i in range(len(t)):
        if t[:i] + t[i+1:] == s:
            return True
    return False

def check_delete(s, t):
    for i in range(len(s)):
        if s[:i] + s[i+1:] == t:
            return True
    return False

def check_change(s, t):
    diff = 0
    for i in range(len(s)):
        if s[i] != t[i]:
            diff += 1
    return diff == 1

def check_equal(s, t):
    return s == t

def check_string(s, t):
    return check_insert(s, t) or check_delete(s, t) or check_change(s, t) or check_equal(s, t)

def main():
    n, t = sys.stdin.readline().split()
    n = int(n)
    strings = [sys.stdin.readline().strip() for _ in range(n)]
    indices = [i+1 for i, s in enumerate(strings) if check_string(s, t)]
    print(len(indices))
    print(' '.join(map(str, indices)))

if __name__ == "__main__":
    main()