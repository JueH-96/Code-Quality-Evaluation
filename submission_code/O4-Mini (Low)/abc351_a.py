def main():
    import sys
    data = sys.stdin.read().split()
    # First 9 numbers are A_i, next 8 numbers are B_j
    A = list(map(int, data[:9]))
    B = list(map(int, data[9:17]))
    sumA = sum(A)
    sumB = sum(B)
    # Need x such that sumB + x > sumA -> x > sumA - sumB
    # Minimum integer x is (sumA - sumB + 1)
    print(sumA - sumB + 1)

if __name__ == "__main__":
    main()