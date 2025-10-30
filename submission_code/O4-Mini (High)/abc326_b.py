def main():
    N = int(input().strip())
    for x in range(N, 1000):
        h = x // 100
        t = (x // 10) % 10
        u = x % 10
        if h * t == u:
            print(x)
            return

if __name__ == "__main__":
    main()