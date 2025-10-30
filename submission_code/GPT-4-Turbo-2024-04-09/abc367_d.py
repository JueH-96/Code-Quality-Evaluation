def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:2+N]))
    
    # Calculate prefix sums of steps
    prefix_sums = [0] * (N + 1)
    for i in range(1, N + 1):
        prefix_sums[i] = prefix_sums[i - 1] + A[i - 1]
    
    # Calculate the remainder of prefix sums when divided by M
    remainder_count = [0] * M
    for sum_value in prefix_sums:
        remainder = sum_value % M
        remainder_count[remainder] += 1
    
    # Calculate the number of valid pairs
    result = 0
    for count in remainder_count:
        if count > 1:
            result += count * (count - 1) // 2
    
    print(result)

if __name__ == "__main__":
    main()