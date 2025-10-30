# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    stack = []
    
    for i in range(N):
        ball_size = 2 ** A[i]
        stack.append(ball_size)
        
        while len(stack) > 1 and stack[-1] == stack[-2]:
            last = stack.pop()
            stack[-1] *= 2
    
    print(len(stack))

if __name__ == "__main__":
    main()