# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    B = list(map(int, data[N+1:2*N]))
    
    A.sort()
    B.sort()
    
    # We need to find the smallest x such that:
    # The number of toys that cannot be placed in the existing boxes is at most 1.
    # And the new box can accommodate the largest toy that cannot be placed in the existing boxes.
    
    # First, find the number of toys that cannot be placed in the existing boxes.
    # We need to match each toy to a box, and count how many toys cannot be matched.
    
    # Initialize pointers for A and B
    i = 0
    j = 0
    count = 0
    while i < N and j < N-1:
        if A[i] <= B[j]:
            i += 1
            j += 1
        else:
            j += 1
    
    # The number of toys that cannot be placed in the existing boxes is N - i
    # Because i is the number of toys that can be placed in the existing boxes
    # So, the number of toys that cannot be placed is N - i
    # We need to have N - i <= 1, because we can add one new box
    if N - i > 1:
        print(-1)
        return
    
    # Now, find the smallest x such that the new box can accommodate the largest toy that cannot be placed in the existing boxes
    # The largest toy that cannot be placed is A[-1] if N - i == 1
    if N - i == 1:
        x = A[-1]
    else:
        # All toys can be placed in the existing boxes
        # So, x can be any value, but we need to find the smallest possible x
        # Since no toy needs to be placed in the new box, x can be 1
        x = 1
    
    # Now, we need to ensure that the new box is not used for any toy that can be placed in the existing boxes
    # So, x must be at least the size of the largest toy that cannot be placed in the existing boxes
    # But since we have already determined that N - i <= 1, and if N - i == 1, x must be at least A[-1]
    # So, x is already determined
    
    # Now, check if the new box can be used to place the toy that cannot be placed in the existing boxes
    # Since x is at least the size of that toy, it can be placed in the new box
    
    print(x)

if __name__ == "__main__":
    main()