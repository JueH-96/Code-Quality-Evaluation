def main():
    import sys
    
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))

    MAX_VAL = 10**6
    
    # Frequency array
    freq = [0] * (MAX_VAL + 1)
    for val in A:
        freq[val] += 1
    
    # Prefix sum of values (not just freq, but value * freq)
    # S[i] will store the sum of all elements in A that are <= i
    S = [0] * (MAX_VAL + 1)
    
    # Build prefix sum: S[i] = S[i-1] + i * freq[i]
    for i in range(1, MAX_VAL + 1):
        S[i] = S[i-1] + i * freq[i]
    
    total_sum = S[MAX_VAL]  # sum of all elements in A
    result = []
    for val in A:
        # Sum of elements greater than val = total_sum - sum of elements <= val
        sum_greater = total_sum - S[val]
        result.append(str(sum_greater))
    
    print(" ".join(result))

# Do not forget to call main()!
if __name__ == "__main__":
    main()