def main():
    import sys
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()
    
    X = []
    while S != T:
        # Find positions where S differs from T
        diff_positions = [i for i in range(len(S)) if S[i] != T[i]]
        # Generate candidates by changing one differing position to T's character
        candidates = []
        for pos in diff_positions:
            # Create new string by changing S[pos] to T[pos]
            new_S = S[:pos] + T[pos] + S[pos+1:]
            candidates.append(new_S)
        # Choose the lex smallest candidate
        next_S = min(candidates)
        X.append(next_S)
        S = next_S
    # Output
    print(len(X))
    for s in X:
        print(s)

if __name__ == "__main__":
    main()