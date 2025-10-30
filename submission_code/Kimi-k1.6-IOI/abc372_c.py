def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    Q = int(data[idx])
    idx += 1
    S = list(data[idx])
    idx += 1
    
    count = 0
    for i in range(len(S) - 2):
        if S[i] == 'A' and S[i+1] == 'B' and S[i+2] == 'C':
            count += 1
    
    for _ in range(Q):
        X = int(data[idx]) - 1  # convert to 0-based
        idx += 1
        C = data[idx]
        idx += 1
        
        # Calculate old contributions
        old = 0
        for s in [X-2, X-1, X]:
            if 0 <= s <= len(S) - 3:
                if S[s] == 'A' and S[s+1] == 'B' and S[s+2] == 'C':
                    old += 1
        
        # Update the character
        S[X] = C
        
        # Calculate new contributions
        new = 0
        for s in [X-2, X-1, X]:
            if 0 <= s <= len(S) - 3:
                if S[s] == 'A' and S[s+1] == 'B' and S[s+2] == 'C':
                    new += 1
        
        count += (new - old)
        print(count)
        
if __name__ == '__main__':
    main()