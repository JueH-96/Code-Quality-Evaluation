def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    N = int(data[idx])
    idx +=1
    Q = int(data[idx])
    idx +=1
    S = list(data[idx])
    idx +=1
    
    count = 0
    for s in range(len(S)-2):
        if S[s] == 'A' and S[s+1] == 'B' and S[s+2] == 'C':
            count +=1
    
    for _ in range(Q):
        X = int(data[idx])
        idx +=1
        C = data[idx]
        idx +=1
        pos = X-1  # convert to 0-based
        
        to_check = [pos-2, pos-1, pos]
        # Subtract before update
        for s in to_check:
            if 0 <= s <= len(S)-3:
                if S[s] == 'A' and S[s+1] == 'B' and S[s+2] == 'C':
                    count -=1
        # Update
        S[pos] = C
        # Add after update
        for s in to_check:
            if 0 <= s <= len(S)-3:
                if S[s] == 'A' and S[s+1] == 'B' and S[s+2] == 'C':
                    count +=1
        print(count)
        
if __name__ == "__main__":
    main()