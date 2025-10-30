def main():
    S = input().strip()
    
    mapping = {')': '(', ']': '[', '>': '<'}
    stack = []
    for c in S:
        if c in '([<':
            stack.append(c)
        else:
            if not stack:
                print("No")
                return
            top = stack.pop()
            if top != mapping[c]:
                print("No")
                return
                
    print("Yes" if not stack else "No")

if __name__ == "__main__":
    main()