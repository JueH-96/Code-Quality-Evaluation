def main():
    S = input().strip()
    
    # If the first two characters are the same, that's our majority character
    if S[0] == S[1]:
        majority_char = S[0]
        for i, ch in enumerate(S):
            if ch != majority_char:
                print(i+1)  # 1-based index
                return
    else:
        # If the first two characters differ, we check the third to see who is the majority
        if S[0] == S[2]:
            # Then S[1] is the outlier
            print(2)
        else:
            # Then S[0] is the outlier
            print(1)

# Do not remove the call to main()
main()