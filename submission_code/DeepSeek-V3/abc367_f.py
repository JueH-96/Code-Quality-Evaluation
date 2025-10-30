# YOUR CODE HERE
import sys

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    Q = int(data[idx+1])
    idx += 2
    A = list(map(int, data[idx:idx+N]))
    idx += N
    B = list(map(int, data[idx:idx+N]))
    idx += N
    
    # Precompute prefix sums for A and B
    # We will use a list of dictionaries to store the frequency of each number up to each index
    # For A
    prefix_A = [{} for _ in range(N+1)]
    for i in range(1, N+1):
        prefix_A[i] = prefix_A[i-1].copy()
        if A[i-1] in prefix_A[i]:
            prefix_A[i][A[i-1]] += 1
        else:
            prefix_A[i][A[i-1]] = 1
    
    # For B
    prefix_B = [{} for _ in range(N+1)]
    for i in range(1, N+1):
        prefix_B[i] = prefix_B[i-1].copy()
        if B[i-1] in prefix_B[i]:
            prefix_B[i][B[i-1]] += 1
        else:
            prefix_B[i][B[i-1]] = 1
    
    # Process queries
    for _ in range(Q):
        l = int(data[idx])
        r = int(data[idx+1])
        L = int(data[idx+2])
        R = int(data[idx+3])
        idx += 4
        
        # Get the frequency dictionary for A[l..r]
        freq_A = {}
        for key in prefix_A[r]:
            if key in prefix_A[l-1]:
                freq_A[key] = prefix_A[r][key] - prefix_A[l-1][key]
            else:
                freq_A[key] = prefix_A[r][key]
        
        # Get the frequency dictionary for B[L..R]
        freq_B = {}
        for key in prefix_B[R]:
            if key in prefix_B[L-1]:
                freq_B[key] = prefix_B[R][key] - prefix_B[L-1][key]
            else:
                freq_B[key] = prefix_B[R][key]
        
        # Compare the two frequency dictionaries
        if freq_A == freq_B:
            print("Yes")
        else:
            print("No")

if __name__ == "__main__":
    main()