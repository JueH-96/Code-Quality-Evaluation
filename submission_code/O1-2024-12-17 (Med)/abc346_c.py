def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, K = map(int, data[:2])
    A = list(map(int, data[2:]))

    unique_in_range = set()  
    for num in A:
        if 1 <= num <= K:
            unique_in_range.add(num)

    total_sum = K * (K + 1) // 2  # sum of 1 to K
    sum_in_sequence = sum(unique_in_range)
    result = total_sum - sum_in_sequence
    print(result)

if __name__ == "__main__":
    main()