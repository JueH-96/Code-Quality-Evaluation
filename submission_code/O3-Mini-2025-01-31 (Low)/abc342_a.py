def main():
    import sys
    input = sys.stdin.readline
    S = input().strip()
    
    # Check for the earliest ambiguous case.
    if S[0] != S[1]:
        # If S[0] equals S[2], then S[1] is different.
        if S[0] == S[2]:
            print(2)
        else:
            # Otherwise, S[0] must be the different one.
            print(1)
    else:
        # S[0] and S[1] are same, so they are the common character.
        common = S[0]
        # Look for the character that is different.
        for i in range(2, len(S)):
            if S[i] != common:
                print(i+1)  # 1-indexed result
                break

if __name__ == "__main__":
    main()