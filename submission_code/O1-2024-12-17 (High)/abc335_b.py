def main():
    import sys
    N = int(sys.stdin.read().strip())
    for x in range(N + 1):
        for y in range(N + 1 - x):
            for z in range(N + 1 - x - y):
                print(x, y, z)

if __name__ == "__main__":
    main()