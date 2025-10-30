def main():
    import sys
    input = sys.stdin.read
    S = input().strip()
    
    substrings = set()
    
    # Generate all possible substrings
    for i in range(len(S)):
        for j in range(i + 1, len(S) + 1):
            substrings.add(S[i:j])
    
    # Output the number of unique substrings
    print(len(substrings))

if __name__ == "__main__":
    main()