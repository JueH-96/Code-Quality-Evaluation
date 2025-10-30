def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N, K = map(int, data[:2])
    A = list(map(int, data[2:]))

    # Filter out elements that are greater than K (they don't matter for 1..K)
    filtered = [x for x in A if x <= K]

    # Get distinct elements (convert to set)
    distinct_elements = set(filtered)

    # Sum of numbers from 1 to K
    total_sum = K * (K + 1) // 2

    # Subtract the sum of distinct elements that appear in A
    answer = total_sum - sum(distinct_elements)

    print(answer)

def main():
    solve()

if __name__ == "__main__":
    main()