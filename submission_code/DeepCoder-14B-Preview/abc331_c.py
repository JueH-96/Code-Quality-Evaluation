import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    A = list(map(int, input[1:n+1]))
    
    sorted_A = sorted(A)
    n_sorted = len(sorted_A)
    
    # Compute suffix sums
    suffix_sum = [0] * (n_sorted + 1)
    current_sum = 0
    for i in range(n_sorted - 1, -1, -1):
        current_sum += sorted_A[i]
        suffix_sum[i] = current_sum
    
    # Calculate results for each element
    result = []
    for x in A:
        pos = bisect.bisect_right(sorted_A, x)
        result.append(str(suffix_sum[pos]))
    
    print(' '.join(result))

if __name__ == "__main__":
    main()