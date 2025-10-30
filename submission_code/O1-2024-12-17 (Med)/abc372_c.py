def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, Q = map(int, data[:2])
    S = list(data[2])
    
    # Function to check if S[i:i+3] == "ABC"
    def is_abc(i):
        return (i + 2 < N and S[i] == 'A' and S[i+1] == 'B' and S[i+2] == 'C')
    
    # Precompute matches: matches[i] = 1 if S[i...i+2] == "ABC" else 0
    matches = [0] * (N - 2) if N >= 3 else []
    for i in range(N - 2):
        matches[i] = 1 if is_abc(i) else 0
    
    total = sum(matches)
    
    idx = 3
    answers = []
    for _ in range(Q):
        X = int(data[idx]); idx += 1
        C = data[idx]; idx += 1
        pos = X - 1  # convert to 0-based index
        
        # Update only if there's an actual change
        if S[pos] != C:
            S[pos] = C  # update character in the string
            # Recalculate up to 3 potential "ABC" overlaps
            for i in range(pos - 2, pos + 1):
                if 0 <= i <= N - 3:
                    old_val = matches[i]
                    new_val = 1 if is_abc(i) else 0
                    matches[i] = new_val
                    total += (new_val - old_val)
        
        answers.append(str(total))
    
    print("
".join(answers))

# Do not forget to call main()
main()