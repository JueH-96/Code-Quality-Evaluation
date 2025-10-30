import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    A = list(map(int, data[1:]))

    stack = []
    for a in A:
        stack.append(a)
        # As long as the top two balls have the same exponent, merge them
        while len(stack) >= 2 and stack[-1] == stack[-2]:
            top = stack.pop()
            stack.pop()
            stack.append(top + 1)
    # The number of balls remaining is the stack size
    print(len(stack))

if __name__ == "__main__":
    main()