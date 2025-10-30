def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    operations = 0
    while True:
        # Sort in descending order
        A.sort(reverse=True)
        
        # Count how many elements are > 0
        positives = sum(1 for x in A if x > 0)
        
        # If fewer than 2 are positive, stop
        if positives < 2:
            break
        
        # Decrease the top two elements by 1
        A[0] -= 1
        A[1] -= 1
        
        operations += 1
    
    print(operations)

# Call the main function
main()