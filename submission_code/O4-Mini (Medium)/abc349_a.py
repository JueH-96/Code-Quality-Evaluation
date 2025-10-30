def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    # First value is N, next N-1 values are A_i
    N = int(data[0])
    A = list(map(int, data[1:1 + (N - 1)]))
    # Since total sum of scores remains 0, A_N = -sum(A_1,...,A_{N-1})
    result = -sum(A)
    sys.stdout.write(str(result))

if __name__ == "__main__":
    main()