def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    a_list = list(map(int, input[1:n+1]))
    
    stack = []
    for a in a_list:
        current = a
        while stack and stack[-1] == current:
            stack.pop()
            current += 1
        stack.append(current)
    print(len(stack))

if __name__ == "__main__":
    main()