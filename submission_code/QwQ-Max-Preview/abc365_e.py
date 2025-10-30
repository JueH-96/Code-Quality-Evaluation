def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    a = list(map(int, input[1:n+1]))
    
    # Compute prefix XOR array
    xor_pre = [0] * (n + 1)
    for i in range(1, n+1):
        xor_pre[i] = xor_pre[i-1] ^ a[i-1]
    
    counts = [0] * 31  # To track the number of set bits for each position
    total = 0
    
    for j in range(2, n+1):
        k = j - 2
        x_k = xor_pre[k]
        
        # Update counts with x_k
        for b in range(31):
            if x_k & (1 << b):
                counts[b] += 1
        
        x_j = xor_pre[j]
        current_sum = 0
        total_elements = k + 1  # Since k = j-2, elements 0..k: k+1 elements
        
        for b in range(31):
            mask = 1 << b
            if x_j & mask:
                cnt = total_elements - counts[b]
            else:
                cnt = counts[b]
            current_sum += cnt * mask
        
        total += current_sum
    
    print(total)

if __name__ == '__main__':
    main()