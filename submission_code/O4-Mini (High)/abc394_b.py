def main():
    import sys
    input = sys.stdin.readline
    n = int(input().strip())
    strings = [input().strip() for _ in range(n)]
    # Sort strings by their length in ascending order
    strings.sort(key=len)
    # Concatenate and print the result
    print(''.join(strings))

if __name__ == '__main__':
    main()