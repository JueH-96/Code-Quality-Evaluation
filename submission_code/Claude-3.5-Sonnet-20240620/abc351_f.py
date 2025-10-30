# YOUR CODE HERE
def solve():
    N = int(input())
    A = list(map(int, input().split()))
    
    result = 0
    stack = []
    
    for i in range(N - 1, -1, -1):
        while stack and A[i] >= stack[-1][0]:
            stack.pop()
        
        if stack:
            result += stack[-1][1] - A[i] * len(stack)
        
        if not stack or A[i] != stack[-1][0]:
            stack.append((A[i], result))
    
    print(result)

solve()