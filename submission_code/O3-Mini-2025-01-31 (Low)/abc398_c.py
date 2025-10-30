def main():
    import sys
    input = sys.stdin.readline
    
    N = int(input().strip())
    A = list(map(int, input().strip().split()))
    
    # Count frequency of each number
    freq = {}
    for a in A:
        freq[a] = freq.get(a, 0) + 1
        
    # Look for candidates: persons with unique values.
    candidate_label = -1  # default result if none exist
    candidate_val = -1
    
    for i in range(N):
        a = A[i]
        if freq[a] == 1:
            # Check if this value is the largest among the candidates
            if a > candidate_val:
                candidate_val = a
                candidate_label = i + 1  # labels are 1-indexed
                
    # Print the final result
    print(candidate_label)

if __name__ == '__main__':
    main()