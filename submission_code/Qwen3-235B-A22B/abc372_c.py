import sys

def main():
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
    for i in range(N - 2):
        if S[i] == 'A' and S[i+1] == 'B' and S[i+2] == 'C':
            count += 1
    
    results = []
    for _ in range(Q):
        X = int(data[idx]) - 1  # convert to 0-based index
        idx += 1
        C = data[idx]
        idx += 1
        
        s_list = []
        for s in [X-2, X-1, X]:
            if 0 <= s <= N - 3:
                s_list.append(s)
        
        # Subtract old contributions
        for s in s_list:
            if S[s] == 'A' and S[s+1] == 'B' and S[s+2] == 'C':
                count -= 1
        
        # Update character
        S[X] = C
        
        # Add new contributions
        for s in s_list:
            if S[s] == 'A' and S[s+1] == 'B' and S[s+2] == 'C':
                count += 1
        
        results.append(str(count))
    
    print('
'.join(results))

if __name__ == "__main__":
    main()