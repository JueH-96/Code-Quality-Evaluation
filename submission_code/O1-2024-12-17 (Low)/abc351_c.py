def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    stack = []
    for power in A:
        stack.append(power)
        # Keep combining while the last two balls have the same power
        while len(stack) >= 2 and stack[-1] == stack[-2]:
            top_power = stack.pop()
            stack.pop()  # remove the second top
            stack.append(top_power + 1)  # new ball with combined power

    print(len(stack))

if __name__ == "__main__":
    main()