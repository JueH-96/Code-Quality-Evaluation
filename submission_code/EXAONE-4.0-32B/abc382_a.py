def main():
    import sys
    data = sys.stdin.read().splitlines()
    n, d = map(int, data[0].split())
    s = data[1].strip()
    total_cookies = s.count('@')
    result = n - total_cookies + d
    print(result)

if __name__ == "__main__":
    main()