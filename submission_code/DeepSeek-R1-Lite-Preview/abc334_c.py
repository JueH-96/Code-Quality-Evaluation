def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:2+K]))
    
    # Create a set for lost colors
    lost = set(A)
    
    # Collect single socks from lost colors
    single_socks = []
    for color in lost:
        single_socks.append(color)
    
    # Determine x: number of same-color pairs to keep
    x = min(N - K, (2 * N - K) // 2)
    
    # Calculate the remaining pairs to make from single socks
    P = (2 * N - K) // 2 - x
    
    # If we need to split some pairs to get more single socks
    if 2 * len(single_socks) < 2 * P:
        # Need to split (P * 2 - len(single_socks)) / 2 pairs
        # But since each split adds 2 single socks, we need to split ceil((2P - len(single_socks)) / 2)
        additional_socks_needed = 2 * P - len(single_socks)
        pairs_to_split = (additional_socks_needed + 1) // 2
        # Choose pairs to split from the colors not in A that are not kept as same-color pairs
        # Since it's optimal to split the pairs that are farthest apart in color to minimize weirdness
        # For simplicity, we assume we can split any pairs not kept as same-color pairs
        # Here, we just add the necessary number of single socks
        single_socks += [i for i in range(1, N+1) if i not in lost][:pairs_to_split*2]
    
    # Sort the single socks
    single_socks.sort()
    
    # Make P pairs from the first 2P single socks
    total_weirdness = 0
    for i in range(P):
        j = 2 * i
        k = 2 * i + 1
        if k < len(single_socks):
            total_weirdness += abs(single_socks[j] - single_socks[k])
    
    print(total_weirdness)

if __name__ == '__main__':
    main()