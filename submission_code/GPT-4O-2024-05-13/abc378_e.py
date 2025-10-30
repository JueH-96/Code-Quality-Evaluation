# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:]))
    
    # Compute prefix sums
    prefix_sums = [0] * (N + 1)
    for i in range(1, N + 1):
        prefix_sums[i] = prefix_sums[i - 1] + A[i - 1]
    
    # Calculate the result
    result = 0
    for l in range(1, N + 1):
        for r in range(l, N + 1):
            subarray_sum = prefix_sums[r] - prefix_sums[l - 1]
            result += subarray_sum % M
    
    print(result)

if __name__ == "__main__":
    main()