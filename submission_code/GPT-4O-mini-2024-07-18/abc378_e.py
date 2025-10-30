def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:2 + N]))
    
    # Prefix sum array
    prefix_sum = [0] * (N + 1)
    
    for i in range(1, N + 1):
        prefix_sum[i] = prefix_sum[i - 1] + A[i - 1]
    
    total_sum = 0
    
    # Calculate the required sum
    for l in range(1, N + 1):
        for r in range(l, N + 1):
            total_sum += (prefix_sum[r] - prefix_sum[l - 1]) % M
    
    print(total_sum)

if __name__ == "__main__":
    main()