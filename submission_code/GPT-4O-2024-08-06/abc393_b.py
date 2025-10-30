# YOUR CODE HERE
def count_abc_triplets(S):
    n = len(S)
    count = 0
    
    # Iterate over possible positions for i
    for i in range(n):
        if S[i] != 'A':
            continue
        
        # Iterate over possible positions for j
        for j in range(i + 1, n):
            if S[j] != 'B':
                continue
            
            # Calculate the position for k
            k = j + (j - i)
            
            # Check if k is within bounds and S[k] is 'C'
            if k < n and S[k] == 'C':
                count += 1
    
    return count

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    S = input().strip()
    result = count_abc_triplets(S)
    print(result)