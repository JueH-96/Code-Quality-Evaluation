def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    B = [A[i] * A[i+1] for i in range(N-1)]
    print(" ".join(map(str, B)))

if __name__ == "__main__":
    main()