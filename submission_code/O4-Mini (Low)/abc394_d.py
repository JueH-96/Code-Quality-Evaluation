import sys

def main():
    S = sys.stdin.readline().strip()
    stack = []
    # mapping of closing to opening
    matches = {')': '(', ']': '[', '>': '<'}
    
    for ch in S:
        if ch in "([<":
            stack.append(ch)
        else:
            # closing bracket
            if not stack or stack[-1] != matches.get(ch, None):
                print("No")
                return
            stack.pop()
    # if stack empty it's balanced
    print("Yes" if not stack else "No")

if __name__ == "__main__":
    main()