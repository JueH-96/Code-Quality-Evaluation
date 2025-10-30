import sys

def main():
    n = int(sys.stdin.readline())
    s = sys.stdin.readline().strip()
    stack = []
    match = [None] * n
    for i in range(n):
        if s[i] == '(':
            stack.append(i)
        elif s[i] == ')':
            if stack:
                l = stack.pop()
                match[l] = i
                match[i] = l
    res = []
    i = 0
    while i < n:
        if s[i] == '(' and match[i] is not None:
            i = match[i] + 1
        else:
            res.append(s[i])
            i += 1
    print(''.join(res))

if __name__ == "__main__":
    main()