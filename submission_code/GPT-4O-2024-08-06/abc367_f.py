# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    # Read N and Q
    index = 0
    N = int(data[index])
    Q = int(data[index + 1])
    index += 2
    
    # Read sequence A
    A = list(map(int, data[index:index + N]))
    index += N
    
    # Read sequence B
    B = list(map(int, data[index:index + N]))
    index += N
    
    # Process each query
    results = []
    for _ in range(Q):
        l_i = int(data[index]) - 1
        r_i = int(data[index + 1]) - 1
        L_i = int(data[index + 2]) - 1
        R_i = int(data[index + 3]) - 1
        index += 4
        
        # Extract subsequences
        sub_A = A[l_i:r_i + 1]
        sub_B = B[L_i:R_i + 1]
        
        # Check if they can be rearranged to match
        if sorted(sub_A) == sorted(sub_B):
            results.append("Yes")
        else:
            results.append("No")
    
    # Print results for each query
    for result in results:
        print(result)

if __name__ == "__main__":
    main()