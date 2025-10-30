def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1
    
    adjacent_pairs = set()
    
    for _ in range(M):
        arr = list(map(int, input[idx:idx+N]))
        idx += N
        for i in range(N - 1):
            a = arr[i]
            b = arr[i+1]
            if a < b:
                pair = (a, b)
            else:
                pair = (b, a)
            adjacent_pairs.add(pair)
    
    total_pairs = N * (N - 1) // 2
    print(total_pairs - len(adjacent_pairs))

if __name__ == "__main__":
    main()