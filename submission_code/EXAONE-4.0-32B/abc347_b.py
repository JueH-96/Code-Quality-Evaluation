def main():
    s = input().strip()
    n = len(s)
    substr_set = set()
    for i in range(n):
        for j in range(i + 1, n + 1):
            substr = s[i:j]
            substr_set.add(substr)
    print(len(substr_set))

if __name__ == "__main__":
    main()