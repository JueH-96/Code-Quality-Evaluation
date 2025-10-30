def main():
    n = int(input().strip())
    seen = set()
    for _ in range(n):
        s = input().strip()
        rev = s[::-1]
        if s <= rev:
            seen.add(s)
        else:
            seen.add(rev)
    print(len(seen))

if __name__ == "__main__":
    main()