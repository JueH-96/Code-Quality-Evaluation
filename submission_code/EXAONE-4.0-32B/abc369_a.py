def main():
    A, B = map(int, input().split())
    candidates = set()
    total = A + B
    if total % 2 == 0:
        candidates.add(total // 2)
    candidates.add(2 * A - B)
    candidates.add(2 * B - A)
    print(len(candidates))

if __name__ == "__main__":
    main()