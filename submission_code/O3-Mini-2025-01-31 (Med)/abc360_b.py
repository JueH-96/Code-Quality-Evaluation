def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    S = data[0]
    T = data[1]
    n = len(S)
    
    # Trying all pairs (c, w) such that 1 <= c <= w and w < len(S)
    for w in range(1, n):
        for c in range(1, w+1):
            # Split S into substrings of length w (except possibly the last substring)
            blocks = [S[i:i+w] for i in range(0, n, w)]
            collected_chars = []
            # For each block, if its length is at least c, append the c-th character (1-indexed)
            for block in blocks:
                if len(block) >= c:
                    collected_chars.append(block[c-1])
            # Check if the concatenation matches T
            if "".join(collected_chars) == T:
                print("Yes")
                return
    print("No")

if __name__ == '__main__':
    main()