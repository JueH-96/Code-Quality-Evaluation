def main():
    s = input().strip()
    closing_to_open = {')': '(', ']': '[', '>': '<'}
    stack = []
    for c in s:
        if c in closing_to_open:
            if not stack:
                print("No")
                return
            top = stack.pop()
            if top != closing_to_open[c]:
                print("No")
                return
        else:
            stack.append(c)
    if not stack:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()