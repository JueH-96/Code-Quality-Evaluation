def main():
    import sys
    data = sys.stdin.read().split()
    N, K = map(int, data[:2])
    A = map(int, data[2:])
    
    # Collect distinct elements from A that are <= K
    distinct_elements = set()
    for x in A:
        if x <= K:
            distinct_elements.add(x)
    
    # Sum of integers from 1 to K
    total_sum_1_to_K = K * (K + 1) // 2
    
    # Subtract the sum of distinct elements in [1..K] to get the result
    answer = total_sum_1_to_K - sum(distinct_elements)
    
    print(answer)

# Do not forget to call main() to be awarded points
if __name__ == "__main__":
    main()