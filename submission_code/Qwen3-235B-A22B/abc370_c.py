def main():
    import sys
    input = sys.stdin.read().split()
    S = input[0]
    T = input[1]
    
    if S == T:
        print(0)
        return
    
    X = []
    current = S
    
    while current != T:
        differing = []
        for i in range(len(current)):
            if current[i] != T[i]:
                differing.append(i)
        # Generate all possible candidates
        candidates = []
        for idx in differing:
            new_s = current[:idx] + T[idx] + current[idx+1:]
            candidates.append(new_s)
        # Find lex smallest candidate
        lex_smallest = min(candidates)
        X.append(lex_smallest)
        current = lex_smallest
    
    print(len(X))
    for line in X:
        print(line)

if __name__ == "__main__":
    main()