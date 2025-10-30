def main():
    import sys
    N = int(sys.stdin.read())
    count = 0
    while N % 2 == 0:
        count += 1
        N //= 2
    print(count)

if __name__ == "__main__":
    main()