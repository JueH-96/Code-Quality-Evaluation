def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    operations = 0
    
    while True:
        # Filter out non-positive elements
        A = [x for x in A if x > 0]
        
        # If there is one or fewer positive elements, we stop
        if len(A) <= 1:
            break
        
        # Sort in descending order
        A.sort(reverse=True)
        
        # Decrease the top two elements
        A[0] -= 1
        A[1] -= 1
        
        # Increment the operation count
        operations += 1
    
    print(operations)

if __name__ == "__main__":
    main()