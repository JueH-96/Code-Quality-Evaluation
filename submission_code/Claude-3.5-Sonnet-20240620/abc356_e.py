# YOUR CODE HERE
def calculate_sum(N, A):
    stack = []
    result = 0
    
    for i in range(N - 1, -1, -1):
        while stack and A[i] > stack[-1][0]:
            _, count = stack.pop()
            result += count * (A[i] // stack[-1][0] if stack else 0)
        
        if stack and A[i] == stack[-1][0]:
            stack[-1][1] += 1
        else:
            stack.append([A[i], 1])
        
        result += sum(A[i] // a * c for a, c in stack[1:])
    
    return result

N = int(input())
A = list(map(int, input().split()))

print(calculate_sum(N, A))