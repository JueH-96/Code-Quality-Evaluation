def main():
    import sys
    S = sys.stdin.read().strip()
    # If there are multiple lines, take the first one.
    if "
" in S:
        S = S.splitlines()[0]
        
    # If the first two characters are the same, then the unique character differs from S[0].
    if S[0] == S[1]:
        base = S[0]
        for i, ch in enumerate(S):
            if ch != base:
                print(i + 1)  # converting 0-index to 1-index
                return
    else:
        # If the first two characters differ, check the third character to resolve which one is unique.
        if S[2] == S[0]:
            print(2)
        else:
            print(1)

if __name__ == "__main__":
    main()