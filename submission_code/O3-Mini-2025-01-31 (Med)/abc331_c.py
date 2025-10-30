def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    # Find the maximum value to size our frequency and prefix arrays.
    max_val = max(A)
    
    # Build a frequency array for values from 0 to max_val.
    freq = [0] * (max_val + 1)
    for value in A:
        freq[value] += 1
    
    # Construct the prefix sum array where prefix[v] is the sum of all elements <= v.
    prefix = [0] * (max_val + 1)
    for v in range(1, max_val + 1):
        prefix[v] = prefix[v - 1] + v * freq[v]
    
    total_sum = prefix[max_val]  # Total sum of all elements in A.
    
    # For each element A[i], compute the sum of all elements greater than A[i].
    # This is given by: total_sum - prefix[A[i]]
    result = []
    for value in A:
        result.append(str(total_sum - prefix[value]))
    
    sys.stdout.write(" ".join(result))

    
if __name__ == '__main__':
    main()