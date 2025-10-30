def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    ptr = 0
    N = int(data[ptr])
    ptr += 1
    Q = int(data[ptr])
    ptr += 1
    
    S = list(data[ptr])
    ptr += 1
    
    # Initial count
    count = 0
    for i in range(N - 2):
        if S[i] == 'A' and S[i+1] == 'B' and S[i+2] == 'C':
            count += 1
    
    for _ in range(Q):
        X_i = int(data[ptr]) - 1  # Convert to 0-based
        ptr += 1
        C = data[ptr]
        ptr += 1
        
        # Iterate over the four possible triplets that include X_i
        for i in [X_i - 2, X_i - 1, X_i, X_i + 1]:
            if 0 <= i < N - 2:
                # Check before update
                if S[i] == 'A' and S[i+1] == 'B' and S[i+2] == 'C':
                    count -= 1
                # Update S
                S[i] = C
                # Check after update
                if S[i] == 'A' and S[i+1] == 'B' and S[i+2] == 'C':
                    count += 1
        
        print(count)

if __name__ == '__main__':
    main()