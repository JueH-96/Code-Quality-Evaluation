def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    stack = []
    for exp in A:
        stack.append(exp)
        # Merge while top two exponents are the same
        while len(stack) >= 2 and stack[-1] == stack[-2]:
            top = stack.pop()
            stack.pop()  # second top
            stack.append(top + 1)

    print(len(stack))

# Do not forget to call main
if __name__ == "__main__":
    main()