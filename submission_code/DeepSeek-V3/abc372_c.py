# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    Q = int(data[1])
    S = list(data[2])
    
    # Precompute the initial count of 'ABC'
    count = 0
    for i in range(N - 2):
        if S[i] == 'A' and S[i+1] == 'B' and S[i+2] == 'C':
            count += 1
    
    for i in range(Q):
        X = int(data[3 + 2*i]) - 1
        C = data[3 + 2*i + 1]
        
        # Check the affected positions
        # The positions that could be part of 'ABC' are X-2, X-1, X, X+1, X+2
        # So we need to check the substrings that include these positions
        
        # First, remove the old contributions
        for j in range(max(0, X-2), min(N-2, X+1)):
            if S[j] == 'A' and S[j+1] == 'B' and S[j+2] == 'C':
                count -= 1
        
        # Update the character
        S[X] = C
        
        # Then, add the new contributions
        for j in range(max(0, X-2), min(N-2, X+1)):
            if S[j] == 'A' and S[j+1] == 'B' and S[j+2] == 'C':
                count += 1
        
        print(count)

if __name__ == "__main__":
    main()