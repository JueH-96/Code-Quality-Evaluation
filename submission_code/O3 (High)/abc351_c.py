import sys

def main() -> None:
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    n = int(data[0])
    a_list = list(map(int, data[1:]))

    stack = []                 # stores exponents of the balls
    append = stack.append
    pop = stack.pop

    for x in a_list:
        append(x)              # add the new ball
        # repeatedly merge while the last two balls have the same size
        while len(stack) >= 2 and stack[-1] == stack[-2]:
            merged_val = pop() + 1   # remove one, add 1 to exponent
            pop()                    # remove the second equal ball
            append(merged_val)       # push the merged ball

    print(len(stack))

if __name__ == "__main__":
    main()