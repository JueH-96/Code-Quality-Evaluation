def main():
    import sys
    data = sys.stdin.read().strip()
    if not data:
        return
    N = int(data)
    # Search from N up to 999 inclusive
    for x in range(N, 1000):
        if 100 <= x <= 999:
            h = x // 100
            t = (x // 10) % 10
            u = x % 10
            if h * t == u:
                print(x)
                return

if __name__ == "__main__":
    main()