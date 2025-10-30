def main():
    import sys
    
    input_data = sys.stdin.read().strip().split()
    N, Q = map(int, input_data[:2])
    S = list(input_data[2])
    
    # Precompute the number of "ABC" occurrences in S
    def is_abc(i):
        return (S[i] == 'A' and S[i+1] == 'B' and S[i+2] == 'C')
    
    total_abc = 0
    for i in range(N-2):
        if is_abc(i):
            total_abc += 1
    
    idx = 3  # starting index in input_data for queries
    
    answers = []
    for _ in range(Q):
        x = int(input_data[idx]) - 1  # convert to 0-based
        c = input_data[idx + 1]
        idx += 2
        
        # For each relevant start index in [x-2, x-1, x], remove old occurrences if any
        for start in range(x-2, x+1):
            if 0 <= start <= N-3:
                if is_abc(start):
                    total_abc -= 1
        
        # Update S[x] to c
        S[x] = c
        
        # Check new occurrences
        for start in range(x-2, x+1):
            if 0 <= start <= N-3:
                if is_abc(start):
                    total_abc += 1
        
        answers.append(str(total_abc))
    
    print("
".join(answers))

# Do not forget to call main
if __name__ == "__main__":
    main()