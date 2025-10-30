# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    S = data[1]
    Q = int(data[2])
    
    operations = data[3:]
    
    # We will use a list to make string modifications easier
    S = list(S)
    
    # Perform each operation
    for i in range(Q):
        c_i = operations[2 * i]
        d_i = operations[2 * i + 1]
        
        # Replace all occurrences of c_i with d_i
        for j in range(N):
            if S[j] == c_i:
                S[j] = d_i
    
    # Convert list back to string
    result = ''.join(S)
    
    # Print the final result
    print(result)

if __name__ == "__main__":
    main()