def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    Q = int(data[1])
    S = list(data[2])
    
    C = 0
    for i in range(N - 2):
        if S[i] == 'A' and S[i+1] == 'B' and S[i+2] == 'C':
            C += 1
    
    index = 3
    for _ in range(Q):
        X_i = int(data[index])
        C_i = data[index + 1]
        index += 2
        pos = X_i - 1
        
        old_char = S[pos]
        S[pos] = C_i
        
        # Check first substring if possible
        if pos - 2 >= 0 and pos + 1 < N:
            old_sub = [S[pos - 2], S[pos - 1], old_char]
            new_sub = [S[pos - 2], S[pos - 1], S[pos]]
            if old_sub == ['A', 'B', 'C']:
                C -= 1
            if new_sub == ['A', 'B', 'C']:
                C += 1
        
        # Check second substring if possible
        if pos - 1 >= 0 and pos + 2 < N:
            old_sub = [S[pos - 1], old_char, S[pos + 1]]
            new_sub = [S[pos - 1], S[pos], S[pos + 1]]
            if old_sub == ['A', 'B', 'C']:
                C -= 1
            if new_sub == ['A', 'B', 'C']:
                C += 1
        
        print(C)

if __name__ == "__main__":
    main()