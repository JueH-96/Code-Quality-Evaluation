def main():
    import sys
    input = sys.stdin.readline
    
    S = input().strip()
    
    # Determine the majority character.
    if S[0] == S[1]:
        majority = S[0]
    else:
        # When S[0] != S[1], one of them must be the unique.
        # Check S[2] to see which one matches:
        if S[0] == S[2]:
            majority = S[0]
        else:
            majority = S[1]
            
    # Find the index of the unique character.
    for idx, ch in enumerate(S):
        if ch != majority:
            print(idx + 1)  # Convert zero-indexed to one-indexed.
            break

if __name__ == '__main__':
    main()