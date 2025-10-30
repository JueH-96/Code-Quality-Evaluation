def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    prefix = [0] * (N + 1)
    for i in range(1, N + 1):
        prefix[i] = prefix[i-1] ^ A[i-1]
    
    sum_total = 0
    
    for k in range(31):
        bit_mask = 1 << k
        bit_count = 0
        for j in range(1, N + 1):
            if (prefix[j] >> k) & 1:
                bit_count += 1
        
        for j in range(1, N + 1):
            current_bit = (prefix[j] >> k) & 1
            if current_bit:
                contrib = (bit_count - 1) * (1 << k)
            else:
                contrib = ( (j - 1) - (bit_count - 1) ) * (1 << k)
            sum_total += contrib
    
    print(sum_total)

if __name__ == '__main__':
    main()