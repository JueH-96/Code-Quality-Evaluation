import sys

def main():
    S = sys.stdin.readline().strip()
    mapping = {')': '(', ']': '[', '>': '<'}
    stack = []
    for c in S:
        if c in mapping:
            if not stack or stack[-1] != mapping[c]:
                print("No")
                return
            stack.pop()
        else:
            stack.append(c)
    print("Yes" if not stack else "No")

if __name__ == "__main__":
    main()