import sys

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    Q = int(data[1])
    S = list(data[2])
    queries = data[3:]
    
    def count_initial_ABC(S):
        count = 0
        for i in range(len(S)-2):
            if S[i] == 'A' and S[i+1] == 'B' and S[i+2] == 'C':
                count +=1
        return count
    
    total = count_initial_ABC(S)
    
    for i in range(Q):
        X = int(queries[2*i]) -1  # 0-based index
        C = queries[2*i+1]
        
        # Positions to check before replacement: X-2, X-1, X
        for j in range(X-2, X+1):
            if 0 <= j <= N-3:
                if S[j] == 'A' and S[j+1] == 'B' and S[j+2] == 'C':
                    total -=1
        # Replace the character
        S[X] = C
        # Positions to check after replacement: X-2, X-1, X
        for j in range(X-2, X+1):
            if 0 <= j <= N-3:
                if S[j] == 'A' and S[j+1] == 'B' and S[j+2] == 'C':
                    total +=1
        # Print the current total
        print(total)

if __name__ == "__main__":
    main()