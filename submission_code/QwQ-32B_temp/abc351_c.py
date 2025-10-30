import sys

def main():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
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