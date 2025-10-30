def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    Q = int(data[1])
    S = list(data[2])
    
    # Initialize the count of 'ABC' substrings
    count = 0
    for i in range(N - 2):
        if S[i] == 'A' and S[i+1] == 'B' and S[i+2] == 'C':
            count += 1
    
    for i in range(Q):
        X = int(data[3 + 2*i]) - 1
        C = data[3 + 2*i + 1]
        
        # Check the three possible windows that include the X-th character
        # Window 1: X-2, X-1, X
        if X >= 2:
            if S[X-2] == 'A' and S[X-1] == 'B' and S[X] == 'C':
                count -= 1
        # Window 2: X-1, X, X+1
        if X >= 1 and X < N - 1:
            if S[X-1] == 'A' and S[X] == 'B' and S[X+1] == 'C':
                count -= 1
        # Window 3: X, X+1, X+2
        if X < N - 2:
            if S[X] == 'A' and S[X+1] == 'B' and S[X+2] == 'C':
                count -= 1
        
        # Update the character
        S[X] = C
        
        # Recheck the three windows
        # Window 1: X-2, X-1, X
        if X >= 2:
            if S[X-2] == 'A' and S[X-1] == 'B' and S[X] == 'C':
                count += 1
        # Window 2: X-1, X, X+1
        if X >= 1 and X < N - 1:
            if S[X-1] == 'A' and S[X] == 'B' and S[X+1] == 'C':
                count += 1
        # Window 3: X, X+1, X+2
        if X < N - 2:
            if S[X] == 'A' and S[X+1] == 'B' and S[X+2] == 'C':
                count += 1
        
        print(count)

if __name__ == "__main__":
    main()