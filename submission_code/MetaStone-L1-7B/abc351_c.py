def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    
    stack = []
    for a in A:
        stack.append((a, 1))
        while len(stack) >= 2 and stack[-1][0] == stack[-2][0]:
            v = stack.pop()
            prev_v = stack.pop()
            stack.append((prev_v[0] + 1, 1))
    
    print(len(stack))

if __name__ == '__main__':
    main()