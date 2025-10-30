def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:]))
    sum_A = sum(A)
    
    count = [0] * 31
    result_all = 0
    prefix_xor = 0
    total_processed = 0
    
    # Process prefix_xor[0]
    current_val = prefix_xor
    for b in range(31):
        current_bit = (current_val >> b) & 1
        if current_bit == 0:
            different = count[b]
        else:
            different = total_processed - count[b]
        result_all += different * (1 << b)
    # Update count
    for b in range(31):
        count[b] += (current_val >> b) & 1
    total_processed += 1
    
    # Process the rest
    for a in A:
        prefix_xor ^= a
        current_val = prefix_xor
        for b in range(31):
            current_bit = (current_val >> b) & 1
            if current_bit == 0:
                different = count[b]
            else:
                different = total_processed - count[b]
            result_all += different * (1 << b)
        # Update count
        for b in range(31):
            count[b] += (current_val >> b) & 1
        total_processed += 1
    
    result = result_all - sum_A
    print(result)

if __name__ == '__main__':
    main()