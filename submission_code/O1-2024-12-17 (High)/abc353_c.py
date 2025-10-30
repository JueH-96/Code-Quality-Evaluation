def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    MOD_BOUND = 10**8
    sumA = sum(A)
    total = (N - 1) * sumA  # sum over all pairs (A_i + A_j)

    A.sort()
    left, right = 0, N - 1
    count = 0  # number of pairs (i,j) with A[i] + A[j] >= 10^8

    while left < right:
        if A[left] + A[right] >= MOD_BOUND:
            count += (right - left)
            right -= 1
        else:
            left += 1

    # Each pair >= 10^8 contributes an extra 10^8 to subtract
    result = total - count * MOD_BOUND
    print(result)

# Don't forget to call main!
if __name__ == "__main__":
    main()