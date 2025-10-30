def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    unique = set()
    for s in input[1:n+1]:
        rev = s[::-1]
        unique.add(min(s, rev))
    print(len(unique))

if __name__ == "__main__":
    main()