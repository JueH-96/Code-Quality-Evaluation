def main():
    import sys
    input = sys.stdin.readline
    N = int(input().strip())
    strings = [input().strip() for _ in range(N)]
    # sort the string list by their length
    strings.sort(key=len)
    # concatenate and print the result
    print(''.join(strings))

if __name__ == '__main__':
    main()