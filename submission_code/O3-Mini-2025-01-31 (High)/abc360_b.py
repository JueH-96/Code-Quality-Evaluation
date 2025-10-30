def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    S = data[0]
    T = data[1]
    n = len(S)
    
    # Iterate over all possible w where 1 <= w < len(S)
    for w in range(1, n):
        # Iterate c from 1 to w (inclusive)
        for c in range(1, w + 1):
            result = []
            # Split S into substrings of length w (except possibly the last one)
            for i in range(0, n, w):
                chunk = S[i:i + w]
                # Only if chunk has at least c characters, we take the c-th one (1-indexed)
                if len(chunk) >= c:
                    result.append(chunk[c - 1])
            # Check if the concatenation equals T
            if "".join(result) == T:
                print("Yes")
                return
    print("No")

if __name__ == '__main__':
    main()