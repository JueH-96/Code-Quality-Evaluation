def main():
    import sys
    S = sys.stdin.read().strip()
    # We only need the first line if extra whitespace exists
    if "
" in S:
        S = S.split('
')[0].strip()
    
    unique_substrings = set()
    n = len(S)
    for i in range(n):
        for j in range(i+1, n+1):
            unique_substrings.add(S[i:j])
    
    print(len(unique_substrings))

if __name__ == '__main__':
    main()