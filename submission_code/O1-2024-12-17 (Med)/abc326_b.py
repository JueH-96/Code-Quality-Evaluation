def main():
    import sys
    N = int(sys.stdin.readline().strip())

    def is_326_like(x):
        h = x // 100
        t = (x // 10) % 10
        o = x % 10
        return (h * t) == o

    for num in range(N, 1000):
        if is_326_like(num):
            print(num)
            break

if __name__ == "__main__":
    main()