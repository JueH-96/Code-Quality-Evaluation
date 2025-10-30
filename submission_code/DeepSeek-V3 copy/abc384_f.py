# YOUR CODE HERE
import sys

def f(x):
    while x % 2 == 0:
        x = x // 2
    return x

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    # Precompute f(x) for all possible sums
    # Since A_i <= 1e7, the maximum sum is 2e7
    max_sum = 2 * 10**7
    f_values = [0] * (max_sum + 1)
    for x in range(1, max_sum + 1):
        f_values[x] = f(x)
    
    # Count the frequency of each A_i
    from collections import defaultdict
    freq = defaultdict(int)
    for num in A:
        freq[num] += 1
    
    # Calculate the sum
    total = 0
    # Iterate over all unique pairs (i, j) where i <= j
    # To avoid double counting, we can handle i == j separately
    # First, handle i == j
    for num in freq:
        count = freq[num]
        if count >= 1:
            s = num + num
            if s <= max_sum:
                total += count * (count + 1) // 2 * f_values[s]
    
    # Now, handle i < j
    # To optimize, we can iterate over all unique pairs (a, b) where a < b
    # and multiply by the product of their frequencies
    unique_A = sorted(freq.keys())
    for i in range(len(unique_A)):
        a = unique_A[i]
        for j in range(i+1, len(unique_A)):
            b = unique_A[j]
            s = a + b
            if s <= max_sum:
                total += freq[a] * freq[b] * f_values[s]
    
    print(total)

if __name__ == "__main__":
    main()