def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    a = list(map(int, input[1:n+1]))
    stack = []
    for val in a:
        stack.append(val)
        while len(stack) >= 2 and stack[-1] == stack[-2]:
            v = stack.pop()
            stack.pop()
            stack.append(v + 1)
    print(len(stack))

if __name__ == "__main__":
    main()