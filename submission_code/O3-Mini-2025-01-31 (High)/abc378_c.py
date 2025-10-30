def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    A = list(map(int, data[1:]))
    
    # Dictionary to store the last index (1-indexed) where each number occurred.
    last_occurrence = {}
    
    # List to store the results
    B = []
    
    for i in range(n):
        number = A[i]
        # If number has been seen before, B[i] is the last occurred index.
        if number in last_occurrence:
            B.append(last_occurrence[number])
        else:
            B.append(-1)
        # Update last occurrence with the current index (using 1-indexing)
        last_occurrence[number] = i + 1
        
    # Print the result as a space-separated string.
    print(" ".join(map(str, B)))

if __name__ == '__main__':
    main()