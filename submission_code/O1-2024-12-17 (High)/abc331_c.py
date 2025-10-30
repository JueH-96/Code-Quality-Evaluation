def main():
    import sys
    
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))

    max_val = max(A)
    
    # Frequency array
    freq = [0] * (max_val + 1)
    for x in A:
        freq[x] += 1
    
    # Prefix sum of values (valPrefix[x] = sum of k*freq[k] for k <= x)
    valPrefix = [0] * (max_val + 1)
    for i in range(1, max_val + 1):
        valPrefix[i] = valPrefix[i - 1] + i * freq[i]
    
    total_sum = valPrefix[max_val]
    
    # For each A_i, the sum of elements > A_i is total_sum - valPrefix[A_i]
    result = []
    for x in A:
        result.append(total_sum - valPrefix[x])
    
    print(" ".join(map(str, result)))

# Do not forget to call main().
main()