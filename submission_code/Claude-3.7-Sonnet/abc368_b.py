def main():
    # Read input
    N = int(input())
    A = list(map(int, input().split()))
    
    # Count operations
    operations = 0
    
    # Continue until there's at most one positive element
    while sum(1 for a in A if a > 0) > 1:
        # Sort in descending order
        A.sort(reverse=True)
        
        # Decrease the two largest elements by 1
        A[0] -= 1
        A[1] -= 1
        
        # Increment operation count
        operations += 1
    
    # Output the answer
    print(operations)

if __name__ == "__main__":
    main()