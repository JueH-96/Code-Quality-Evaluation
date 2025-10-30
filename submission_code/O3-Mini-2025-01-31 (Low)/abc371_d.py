def main():
    import sys
    import bisect
    data = sys.stdin.read().split()
    n = int(data[0])
    X = list(map(int, data[1:1+n]))
    P = list(map(int, data[1+n:1+2*n]))
    q = int(data[1+2*n])
    
    # Precompute prefix sum
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i+1] = prefix[i] + P[i]
    
    out_lines = []
    index = 2 + 2 * n
    for _ in range(q):
        L = int(data[index])
        R = int(data[index+1])
        index += 2
        # Find left index where X[i] >= L and right index where X[i] > R
        left_idx = bisect.bisect_left(X, L)
        right_idx = bisect.bisect_right(X, R)
        total = prefix[right_idx] - prefix[left_idx]
        out_lines.append(str(total))
    
    sys.stdout.write("
".join(out_lines))

if __name__ == '__main__':
    main()