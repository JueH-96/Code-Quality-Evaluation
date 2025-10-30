def f(x):
    while x % 2 == 0:
        x = x // 2
    return x

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    # Precompute f(x) for all possible sums
    # Since A_i <= 1e7, the maximum possible sum is 2e7
    max_sum = 2 * 10**7
    f_values = [0] * (max_sum + 1)
    for x in range(1, max_sum + 1):
        f_values[x] = f(x)
    
    # Count the frequency of each A_i
    from collections import defaultdict
    freq = defaultdict(int)
    for a in A:
        freq[a] += 1
    
    # Calculate the sum
    total = 0
    # Iterate over all unique pairs (a, b) where a <= b
    unique_A = sorted(freq.keys())
    for i in range(len(unique_A)):
        a = unique_A[i]
        for j in range(i, len(unique_A)):
            b = unique_A[j]
            s = a + b
            if s > max_sum:
                continue
            if a == b:
                # Number of pairs is C(freq[a], 2) + freq[a] (since i=j)
                # Wait, no. For i <= j, when i == j, it's freq[a] * (freq[a] + 1) / 2
                count = freq[a] * (freq[a] + 1) // 2
            else:
                count = freq[a] * freq[b]
            total += count * f_values[s]
    
    print(total)

if __name__ == "__main__":
    main()