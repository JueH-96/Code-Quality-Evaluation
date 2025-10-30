def main():
    S = input().strip()
    # If the first two characters are the same, that's the common char
    if S[0] == S[1]:
        common = S[0]
        # Find the one character that differs
        for i, c in enumerate(S):
            if c != common:
                print(i + 1)
                return
    else:
        # If S[0] != S[1], check S[2] to see which of the first two is the common
        if S[0] == S[2]:
            # S[1] is unique
            print(2)
        else:
            # S[0] is unique
            print(1)

if __name__ == "__main__":
    main()