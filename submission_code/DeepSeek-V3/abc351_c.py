# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    stack = []
    
    for a in A:
        stack.append(a)
        while len(stack) >= 2:
            if stack[-1] == stack[-2]:
                a1 = stack.pop()
                a2 = stack.pop()
                stack.append(a1 + 1)
            else:
                break
    
    print(len(stack))

if __name__ == "__main__":
    main()