def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    stack = []
    for exponent in A:
        stack.append(exponent)
        # Keep merging while top two exponents are the same
        while len(stack) >= 2 and stack[-1] == stack[-2]:
            stack.pop()              # Remove the top
            stack[-1] += 1          # Increment the new top exponent

    print(len(stack))

# Do not forget to call main()!
main()