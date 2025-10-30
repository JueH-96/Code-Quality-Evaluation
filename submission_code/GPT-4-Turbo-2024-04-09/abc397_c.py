def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    # To store the count of distinct elements in the prefix and suffix
    from collections import defaultdict
    
    # Prefix counts of distinct elements
    prefix_distinct_counts = [0] * N
    suffix_distinct_counts = [0] * N
    
    # Dictionary to count occurrences
    prefix_counts = defaultdict(int)
    suffix_counts = defaultdict(int)
    
    # Fill prefix distinct counts
    distinct_count = 0
    for i in range(N):
        if prefix_counts[A[i]] == 0:
            distinct_count += 1
        prefix_counts[A[i]] += 1
        prefix_distinct_counts[i] = distinct_count
    
    # Fill suffix distinct counts
    distinct_count = 0
    for i in range(N-1, -1, -1):
        if suffix_counts[A[i]] == 0:
            distinct_count += 1
        suffix_counts[A[i]] += 1
        suffix_distinct_counts[i] = distinct_count
    
    # Now calculate the maximum sum of distinct counts for all possible splits
    max_sum_distinct = 0
    for i in range(N-1):
        max_sum_distinct = max(max_sum_distinct, prefix_distinct_counts[i] + suffix_distinct_counts[i+1])
    
    print(max_sum_distinct)

if __name__ == "__main__":
    main()