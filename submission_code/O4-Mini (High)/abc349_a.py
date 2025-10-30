def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    # Next N-1 values are A_1 ... A_{N-1}
    A = list(map(int, data[1:1+N-1]))
    # Since the total sum of scores remains zero, A_N = -sum(A_1 ... A_{N-1})
    result = -sum(A)
    print(result)

if __name__ == "__main__":
    main()