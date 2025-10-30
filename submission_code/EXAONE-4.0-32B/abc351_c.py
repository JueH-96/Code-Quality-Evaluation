def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    arr = list(map(int, data[1:1+n]))
    
    stack = []
    for a in arr:
        stack.append(a)
        while len(stack) >= 2 and stack[-1] == stack[-2]:
            top = stack.pop()
            stack.pop()
            stack.append(top + 1)
            
    print(len(stack))

if __name__ == "__main__":
    main()