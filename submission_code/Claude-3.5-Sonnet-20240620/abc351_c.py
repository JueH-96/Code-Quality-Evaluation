# YOUR CODE HERE
def solve():
    N = int(input())
    A = list(map(int, input().split()))
    
    stack = []
    for size in A:
        while len(stack) >= 2 and stack[-1] == stack[-2]:
            stack.pop()
            stack[-1] += 1
        stack.append(size)
    
    while len(stack) >= 2 and stack[-1] == stack[-2]:
        stack.pop()
        stack[-1] += 1
    
    print(len(stack))

solve()