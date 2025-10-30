def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    # Create a frequency array for the values in A
    max_value = 10**6
    freq = [0] * (max_value + 1)
    
    for number in A:
        freq[number] += 1
    
    # Create a prefix sum array to calculate the sum of elements greater than a certain value
    prefix_sum = [0] * (max_value + 1)
    
    for i in range(max_value - 1, -1, -1):
        prefix_sum[i] = prefix_sum[i + 1] + freq[i + 1] * (i + 1)
    
    # Calculate the result for each A[i]
    result = []
    for number in A:
        result.append(prefix_sum[number])
    
    print(" ".join(map(str, result)))

if __name__ == "__main__":
    main()