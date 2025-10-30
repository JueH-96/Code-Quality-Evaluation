def main():
    import sys
    data = sys.stdin.read().split()
    # First 9 numbers are A_i, next 8 are B_j
    A = list(map(int, data[:9]))
    B = list(map(int, data[9:17]))
    sumA = sum(A)
    sumB = sum(B)
    # To win, B must end with strictly more runs: sumB + x > sumA
    # Minimum x is (sumA - sumB + 1)
    x = sumA - sumB + 1
    print(x)

if __name__ == "__main__":
    main()