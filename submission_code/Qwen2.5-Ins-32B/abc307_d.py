import sys

def solve():
    N = int(input())
    S = input().strip()
    stack = []
    result = []
    
    for char in S:
        if char == '(':
            stack.append(len(result))
        elif char == ')' and stack:
            stack.pop()
        else:
            result.append(char)
    
    print(''.join(result))

if __name__ == "__main__":
    solve()