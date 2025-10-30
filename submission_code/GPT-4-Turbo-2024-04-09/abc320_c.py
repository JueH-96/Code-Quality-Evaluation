def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    M = int(data[0])
    S1 = data[1]
    S2 = data[2]
    S3 = data[3]
    
    # We need to find the minimum t such that:
    # S1[(t % M)] == S2[(t % M)] == S3[(t % M)]
    
    # We will simulate the process for each possible t from 0 to 3*M-1
    # Why 3*M-1? Because each string has a period of M, and we need to check
    # at least up to the point where all offsets have been checked against each other.
    
    # We will use a dictionary to store the first occurrence of each character at each position
    # for each string.
    first_occurrence = {}
    
    # We will check up to 3*M to ensure we cover all possible alignments of the cyclic strings
    for t in range(3 * M):
        idx1 = t % M
        idx2 = t % M
        idx3 = t % M
        
        char1 = S1[idx1]
        char2 = S2[idx2]
        char3 = S3[idx3]
        
        # Check if all three characters are the same
        if char1 == char2 == char3:
            print(t)
            return
    
    # If no such t is found, print -1
    print(-1)

if __name__ == "__main__":
    main()