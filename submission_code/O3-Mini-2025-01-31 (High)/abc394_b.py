def main():
    import sys
    input = sys.stdin.readline
    n = int(input().strip())
    strings = [input().strip() for _ in range(n)]
    # Sort strings by their length in ascending order
    strings.sort(key=len)
    # Concatenate sorted strings
    result = "".join(strings)
    print(result)

if __name__ == '__main__':
    main()