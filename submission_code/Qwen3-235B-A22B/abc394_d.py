import sys

def main():
    s = sys.stdin.read().strip()
    if len(s) % 2 != 0:
        print("No")
        return
    stack = []
    for c in s:
        if c in '([{<':
            stack.append(c)
        else:
            if not stack:
                print("No")
                return
            top = stack.pop()
            if (c == ')' and top != '(') or (c == ']' and top != '[') or (c == '>' and top != '<'):
                print("No")
                return
    if not stack:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()