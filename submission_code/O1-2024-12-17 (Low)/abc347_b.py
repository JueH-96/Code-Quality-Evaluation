def main():
    import sys
    S = sys.stdin.readline().strip()
    substrings = set()
    n = len(S)
    
    for i in range(n):
        for j in range(i+1, n+1):
            substrings.add(S[i:j])
    
    print(len(substrings))

if __name__ == "__main__":
    main()