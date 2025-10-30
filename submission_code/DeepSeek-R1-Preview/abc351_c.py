def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    a = list(map(int, input[1:n+1]))
    
    stack = []
    for num in a:
        stack.append(num)
        while True:
            if len(stack) < 2:
                break
            x = stack[-1]
            y = stack[-2]
            if x != y:
                break
            stack.pop()
            stack.pop()
            stack.append(x + 1)
    
    print(len(stack))

if __name__ == "__main__":
    main()