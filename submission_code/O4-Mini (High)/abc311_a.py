def main():
    import sys
    input = sys.stdin.readline

    N = int(input().strip())
    S = input().strip()

    seen = set()
    for i, ch in enumerate(S):
        seen.add(ch)
        if len(seen) == 3:
            print(i + 1)
            return

if __name__ == "__main__":
    main()