def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    
    N = int(input[idx])
    idx += 1
    
    p = []
    for i in range(N):
        a = int(input[idx])
        b = int(input[idx+1])
        idx += 2
        p.append((a, b, i+1))  # (A_i, B_i, original index)
    
    q = []
    for i in range(N):
        c = int(input[idx])
        d = int(input[idx+1])
        idx += 2
        q.append((c, d, i+1))  # (C_i, D_i, original index)
    
    # Sort p and q by (x, y)
    p_sorted = sorted(p, key=lambda x: (x[0], x[1]))
    q_sorted = sorted(q, key=lambda x: (x[0], x[1]))
    
    ans = [0] * N
    for i in range(N):
        p_original_idx = p_sorted[i][2] - 1  # convert to 0-based
        q_original = q_sorted[i][2]
        ans[p_original_idx] = q_original
    
    print(' '.join(map(str, ans)))

if __name__ == '__main__':
    main()