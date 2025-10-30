def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    stack = []

    for e in A:
        stack.append(e)
        # Try to merge while the top two exponents are the same
        while len(stack) > 1 and stack[-1] == stack[-2]:
            top = stack.pop()
            stack.pop()        # remove the matching exponent
            stack.append(top + 1)  # merge into exponent+1

    print(len(stack))

def main():
    solve()

if __name__ == "__main__":
    main()