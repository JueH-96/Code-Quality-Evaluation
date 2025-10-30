def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    
    prefix = [0]
    for num in A:
        prefix.append(prefix[-1] ^ num)
    
    sum_A = sum(A)
    counts = [0] * 31  # bits 0 to 30
    
    for x in prefix:
        for bit in range(31):
            if (x >> bit) & 1:
                counts[bit] += 1
    
    total = len(prefix)
    sum_bits = 0
    for bit in range(31):
        sum_bits += counts[bit] * (total - counts[bit]) * (1 << bit)
    
    print(sum_bits - sum_A)

if __name__ == "__main__":
    main()