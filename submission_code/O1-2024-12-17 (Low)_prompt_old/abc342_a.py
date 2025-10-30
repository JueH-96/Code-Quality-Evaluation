def solve():
    import sys
    S = sys.stdin.readline().strip()
    
    # If the first two chars match, then the majority character is S[0].
    # Find the index of the character which is not S[0].
    if S[0] == S[1]:
        majority_char = S[0]
        for i in range(len(S)):
            if S[i] != majority_char:
                print(i + 1)
                return
    else:
        # Otherwise, compare with the third character
        # If S[2] equals S[0], then S[1] is the odd one out
        # If S[2] equals S[1], then S[0] is the odd one out
        if S[2] == S[0]:
            print(2)
        else:
            print(1)

def main():
    solve()

if __name__ == "__main__":
    main()