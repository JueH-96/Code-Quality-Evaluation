def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:2+K]))
    
    # Initialize sock count
    sock_count = [0] * (N + 1)
    for i in range(1, N + 1):
        if i not in A:
            sock_count[i] = 2
        else:
            sock_count[i] = 1
    
    # Collect single socks
    single_socks = []
    for i in range(1, N + 1):
        if sock_count[i] == 1:
            single_socks.append(i)
    
    # Sort single socks
    single_socks.sort()
    
    # Pair the single socks in order
    total_weirdness = 0
    i = 0
    while i < len(single_socks) - 1:
        # Pair current and next sock
        pair_weirdness = abs(single_socks[i] - single_socks[i + 1])
        total_weirdness += pair_weirdness
        i += 2
    
    print(total_weirdness)

if __name__ == "__main__":
    main()