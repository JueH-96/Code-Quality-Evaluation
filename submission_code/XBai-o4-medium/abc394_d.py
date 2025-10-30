def main():
    s = input().strip()
    stack = []
    matching = {')': '(', ']': '[', '>': '<'}
    for c in s:
        if c in '([<':
            stack.append(c)
        else:
            if not stack:
                print("No")
                return
            top = stack.pop()
            if matching[c] != top:
                print("No")
                return
    print("Yes" if not stack else "No")

if __name__ == "__main__":
    main()