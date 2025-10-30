def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    total_sum = sum(A)
    max_val = max(A)
    
    # Build a frequency array for values from 0 up to max_val.
    freq = [0] * (max_val + 1)
    for a in A:
        freq[a] += 1
        
    # Build a prefix sum array where prefix[x] equals the sum of all elements <= x.
    prefix = [0] * (max_val + 1)
    for x in range(1, max_val + 1):
        prefix[x] = prefix[x-1] + x * freq[x]
    
    # For each number a in A, the answer is total_sum minus the sum of all numbers <= a.
    results = [str(total_sum - prefix[a]) for a in A]
    
    sys.stdout.write(" ".join(results))

if __name__ == '__main__':
    main()