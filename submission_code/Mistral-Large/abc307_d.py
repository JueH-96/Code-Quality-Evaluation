import sys

def remove_nested_parentheses(s):
    stack = []
    result = []

    for char in s:
        if char == '(':
            stack.append(result)
            result = []
        elif char == ')':
            if stack:
                temp = result
                result = stack.pop()
            else:
                result.append(char)
        else:
            result.append(char)

    return ''.join(result)

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    S = data[1]

    result = remove_nested_parentheses(S)
    print(result)

if __name__ == "__main__":
    main()