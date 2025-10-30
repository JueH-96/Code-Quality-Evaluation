import bisect

def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    q = int(data[1+n])
    queries = []
    index = 1 + n + 1
    for i in range(q):
        l = int(data[index])
        r = int(data[index+1])
        index += 2
        queries.append((l, r))
    
    T = [0] * n
    for i in range(1, n):
        if i % 2 == 0:
            T[i] = T[i-1] + (A[i] - A[i-1])
        else:
            T[i] = T[i-1]
    
    out_lines = []
    for (l, r) in queries:
        idx_r = bisect.bisect_right(A, r) - 1
        total_r = T[idx_r]
        if idx_r % 2 == 1:
            total_r += (r - A[idx_r])
        
        idx_l = bisect.bisect_right(A, l) - 1
        total_l = T[idx_l]
        if idx_l % 2 == 1:
            total_l += (l - A[idx_l])
        
        out_lines.append(str(total_r - total_l))
    
    sys.stdout.write("
".join(out_lines))

if __name__ == "__main__":
    main()