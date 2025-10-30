def main():
    import sys
    s = sys.stdin.readline().strip()
    n = len(s)
    seen = set()
    # generate all substrings s[i:j]
    for i in range(n):
        for j in range(i+1, n+1):
            seen.add(s[i:j])
    print(len(seen))

if __name__ == "__main__":
    main()