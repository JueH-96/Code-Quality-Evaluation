# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    # Arrays to store the maximum size each slime can achieve
    left_max = A[:]
    right_max = A[:]
    
    # Left to right pass
    stack = []
    for i in range(N):
        while stack and A[stack[-1]] < A[i]:
            left_max[i] += A[stack.pop()]
        stack.append(i)
    
    # Right to left pass
    stack = []
    for i in range(N-1, -1, -1):
        while stack and A[stack[-1]] < A[i]:
            right_max[i] += A[stack.pop()]
        stack.append(i)
    
    # Calculate the final maximum sizes
    result = [max(left_max[i], right_max[i]) for i in range(N)]
    
    # Print the result
    print(' '.join(map(str, result)))

main()