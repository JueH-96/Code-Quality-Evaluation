def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, X = map(int, data[:2])
    scores = list(map(int, data[2:]))

    total = sum(s for s in scores if s <= X)
    print(total)

main()