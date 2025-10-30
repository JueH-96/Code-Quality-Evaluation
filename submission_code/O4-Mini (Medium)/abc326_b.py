def main():
    import sys
    data = sys.stdin.read().strip()
    if not data:
        return
    N = int(data)
    for x in range(N, 1000):
        h = x // 100
        t = (x // 10) % 10
        o = x % 10
        if h * t == o:
            print(x)
            return

if __name__ == "__main__":
    main()