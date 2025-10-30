def main():
    import sys

    S = sys.stdin.readline().strip()
    
    # Determine the majority character by checking the first three characters
    if S[0] == S[1]:
        majority_char = S[0]
    elif S[1] == S[2]:
        majority_char = S[1]
    else:
        majority_char = S[0]
    
    # Find the index of the character that differs from the majority
    for i, ch in enumerate(S):
        if ch != majority_char:
            print(i + 1)  # 1-based index
            return

# Do not forget to call main()
main()