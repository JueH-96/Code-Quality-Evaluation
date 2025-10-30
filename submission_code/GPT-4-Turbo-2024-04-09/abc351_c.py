def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    stack = []
    
    for a in A:
        power = 2 ** a
        while stack and stack[-1][0] == power:
            count = stack.pop()[1] + 1
            power *= 2
            if stack and stack[-1][0] == power:
                stack[-1][1] += count
            else:
                stack.append([power, count])
                break
        else:
            stack.append([power, 1])
    
    result = sum(item[1] for item in stack)
    print(result)

if __name__ == "__main__":
    main()