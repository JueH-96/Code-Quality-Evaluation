def main():
    import sys
    s = sys.stdin.readline().strip()
    substrings = set()
    n = len(s)
    for i in range(n):
        for j in range(i, n):
            substr = s[i:j+1]
            substrings.add(substr)
    print(len(substrings))

if __name__ == "__main__":
    main()