def main():
    import sys
    N, K = map(int, sys.stdin.readline().split())
    positions = list(range(N*K))
    seq = [0]*(N*K)
    counts = {i: K for i in range(1, N+1)}
    available = list(range(1, N+1))
    
    for _ in range(N*K):
        mid_num_idx = (len(available) - 1) // 2
        num = available[mid_num_idx]
        mid_pos = (len(positions) - 1) // 2
        pos = positions.pop(mid_pos)
        seq[pos] = num
        counts[num] -= 1
        if counts[num] == 0:
            available.remove(num)
    
    print(' '.join(map(str, seq)))

if __name__ == '__main__':
    main()