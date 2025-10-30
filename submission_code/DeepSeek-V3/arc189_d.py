# YOUR CODE HERE
import sys

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    # Precompute the maximum possible size for each position
    # We need to find for each K, the maximum sum Takahashi can achieve by absorbing adjacent smaller slimes
    
    # We will use a stack-based approach to find the maximum sum for each position
    # We will process the array from left to right and then from right to left
    
    # First, compute the sum to the left of each position
    left_sum = [0] * N
    stack = []
    for i in range(N):
        while stack and A[stack[-1]] < A[i]:
            stack.pop()
        if stack:
            left_sum[i] = left_sum[stack[-1]] + A[i]
        else:
            left_sum[i] = A[i]
        stack.append(i)
    
    # Then, compute the sum to the right of each position
    right_sum = [0] * N
    stack = []
    for i in range(N-1, -1, -1):
        while stack and A[stack[-1]] < A[i]:
            stack.pop()
        if stack:
            right_sum[i] = right_sum[stack[-1]] + A[i]
        else:
            right_sum[i] = A[i]
        stack.append(i)
    
    # The maximum size for each K is the sum of the left and right sums minus the original A[K-1] (since it's counted twice)
    B = []
    for i in range(N):
        total = left_sum[i] + right_sum[i] - A[i]
        B.append(str(total))
    
    print(' '.join(B))

if __name__ == "__main__":
    main()