# YOUR CODE HERE
def solve():
    N = int(input())
    H = list(map(int, input().split()))
    
    stack = []
    result = [0] * N
    
    for i in range(N-1, -1, -1):
        while stack and H[stack[-1]] < H[i]:
            stack.pop()
        
        if stack:
            result[i] = stack[-1] - i
        else:
            result[i] = N - i - 1
        
        stack.append(i)
    
    print(*result)

solve()