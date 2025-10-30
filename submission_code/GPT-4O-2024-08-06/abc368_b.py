# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    operation_count = 0
    
    while True:
        # Sort A in descending order
        A.sort(reverse=True)
        
        # Check if we can perform the operation
        if A[0] > 0 and A[1] > 0:
            # Perform the operation
            A[0] -= 1
            A[1] -= 1
            operation_count += 1
        else:
            # If we can't perform the operation, break the loop
            break
    
    print(operation_count)

if __name__ == "__main__":
    main()