def main():
    import sys
    input = sys.stdin.readline
    N = int(input().strip())
    S = input().strip()
    result = ''.join([c * 2 for c in S])
    print(result)

if __name__ == '__main__':
    main()