def main():
    import sys
    n, k = map(int, sys.stdin.readline().split())
    s = sys.stdin.readline().strip()
    parts = s.split('X')
    total = 0
    for seg in parts:
        total += len(seg) // k
    print(total)

if __name__ == "__main__":
    main()