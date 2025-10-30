def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, K = map(int, data[:2])
    A = list(map(int, data[2:]))

    # Calculate the sum of integers from 1 to K
    total_sum = K * (K + 1) // 2

    # Collect distinct integers from A that are ≤ K
    distinct_A = set()
    for num in A:
        if num <= K:
            distinct_A.add(num)

    # Calculate the sum of those distinct integers in [1..K]
    sum_of_distinct = sum(distinct_A)

    # The answer is total_sum minus the sum of the distinct elements
    print(total_sum - sum_of_distinct)

# Do not forget to call main()!
if __name__ == "__main__":
    main()