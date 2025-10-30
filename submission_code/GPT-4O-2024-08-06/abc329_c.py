# YOUR CODE HERE
def count_repeated_substrings(N, S):
    count = 0
    i = 0
    
    while i < N:
        # Start of a new sequence of the same character
        j = i
        while j < N and S[j] == S[i]:
            j += 1
        
        # Length of the sequence of the same character
        length = j - i
        
        # Number of non-empty substrings that can be formed from this sequence
        count += (length * (length + 1)) // 2
        
        # Move to the next different character
        i = j
    
    return count

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    S = data[1]
    result = count_repeated_substrings(N, S)
    print(result)