def main():
    import sys
    input = sys.stdin.readline
    
    S = input().strip()
    T = input().strip()
    
    res = []
    i = 0  # pointer for S
    # iterate over T, find S as a subsequence
    for j, ch in enumerate(T):
        if i < len(S) and ch == S[i]:
            res.append(j + 1)  # Convert 0-based index to 1-based
            i += 1
        # No else: we skip the error characters
        if i == len(S):
            break
            
    print(" ".join(map(str, res)))
    
if __name__ == '__main__':
    main()