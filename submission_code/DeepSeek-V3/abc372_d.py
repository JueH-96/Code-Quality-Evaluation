# YOUR CODE HERE
import sys
import bisect

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    H = list(map(int, data[1:N+1]))
    
    # Initialize the result array
    result = [0] * N
    
    # Use a stack to keep track of the indices of the buildings
    stack = []
    
    # Iterate from the end to the start
    for i in range(N-1, -1, -1):
        # Pop elements from the stack that are smaller than or equal to H[i]
        while stack and H[stack[-1]] <= H[i]:
            stack.pop()
        # The number of valid j's is the number of elements in the stack
        result[i] = len(stack)
        # Push the current index to the stack
        stack.append(i)
    
    # Print the result
    print(' '.join(map(str, result)))

if __name__ == "__main__":
    main()