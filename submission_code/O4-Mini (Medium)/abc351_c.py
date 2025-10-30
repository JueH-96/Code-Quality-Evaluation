import sys

def main():
    import sys
    input = sys.stdin.readline
    N = int(input())
    A = list(map(int, input().split()))
    stack = []
    for a in A:
        stack.append(a)
        # keep merging while the last two balls have the same exponent
        while len(stack) >= 2 and stack[-1] == stack[-2]:
            exp = stack.pop()        # remove top
            stack.pop()              # remove second top
            stack.append(exp + 1)    # push merged ball with exponent+1
    # the remaining balls count is just the stack size
    print(len(stack))

# call main to execute
main()