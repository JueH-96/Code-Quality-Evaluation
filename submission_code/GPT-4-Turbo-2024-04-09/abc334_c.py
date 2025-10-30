def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:2+K]))
    
    # Create a list to count the number of socks of each color
    sock_count = [2] * (N + 1)  # 1-indexed, sock_count[0] is unused
    
    # Mark the lost socks
    for color in A:
        sock_count[color] -= 1
    
    # We need to pair up (2N - K) socks, which means we need to make (2N - K) // 2 pairs
    pairs_needed = (2 * N - K) // 2
    
    # To minimize weirdness, we should pair socks of the same or closest possible colors
    # Greedy approach: try to pair socks of the same color first, then nearest colors
    
    # List to store remaining socks after self-pairing
    remaining_socks = []
    
    # First, pair socks of the same color
    for color in range(1, N + 1):
        if sock_count[color] == 2:
            pairs_needed -= 1
            sock_count[color] -= 2
            if pairs_needed == 0:
                break
        if sock_count[color] > 0:
            remaining_socks.extend([color] * sock_count[color])
    
    # If we still need more pairs, pair the remaining socks
    total_weirdness = 0
    if pairs_needed > 0:
        # We will use a two-pointer technique to pair the remaining socks
        i = 0
        j = len(remaining_socks) - 1
        while pairs_needed > 0 and i < j:
            # Pair sock at i with sock at j
            total_weirdness += abs(remaining_socks[i] - remaining_socks[j])
            i += 1
            j -= 1
            pairs_needed -= 1
    
    print(total_weirdness)

if __name__ == "__main__":
    main()