def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, L = map(int, data[:2])
    A = list(map(int, data[2:]))

    passed_count = sum(1 for score in A if score >= L)
    print(passed_count)

if __name__ == "__main__":
    main()