def main():
    import sys
    n, d = map(int, sys.stdin.readline().split())
    s = sys.stdin.readline().strip()
    initial_empty = s.count('.')
    result = initial_empty + d
    print(result)

if __name__ == '__main__':
    main()