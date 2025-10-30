def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    stack = []
    for a in A:
        stack.append(a)
        while len(stack) >= 2 and stack[-1] == stack[-2]:
            x = stack.pop()
            stack.pop()
            stack.append(x + 1)
    print(len(stack))

if __name__ == "__main__":
    main()