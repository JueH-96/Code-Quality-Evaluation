def main():
    import sys
    S = sys.stdin.readline().strip()
    
    distinct_substrings = set()
    for i in range(len(S)):
        for j in range(i+1, len(S)+1):
            distinct_substrings.add(S[i:j])
    
    print(len(distinct_substrings))

if __name__ == "__main__":
    main()