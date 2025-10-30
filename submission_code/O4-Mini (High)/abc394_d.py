import sys

def main():
    S = sys.stdin.readline().strip()
    stack = []
    # mapping from closing to opening brackets
    match = {')': '(', ']': '[', '>': '<'}
    
    for ch in S:
        if ch in "([<":
            stack.append(ch)
        else:
            # ch is one of ) ] >
            if not stack or stack[-1] != match[ch]:
                print("No")
                return
            stack.pop()
    
    # If stack is empty, all brackets matched
    print("Yes" if not stack else "No")

if __name__ == "__main__":
    main()