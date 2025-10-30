import sys
MOD = 998244353

def main():
    input = sys.stdin.read().split()
    ptr = 0
    H = int(input[ptr])
    ptr += 1
    W = int(input[ptr])
    ptr += 1
    
    max_n = H + W + 2
    fact = [1] * (max_n + 1)
    inv_fact = [1] * (max_n + 1)
    
    for i in range(1, max_n + 1):
        fact[i] = fact[i-1] * i % MOD
    
    inv_fact[max_n] = pow(fact[max_n], MOD-2, MOD)
    for i in range(max_n - 1, -1, -1):
        inv_fact[i] = inv_fact[i + 1] * (i + 1) % MOD
    
    def comb(a, b):
        if a < 0 or b < 0 or a < b:
            return 0
        return fact[a] * inv_fact[b] % MOD * inv_fact[a - b] % MOD
    
    grid = []
    for _ in range(H):
        row = list(map(int, input[ptr:ptr+W]))
        ptr += W
        grid.append(row)
    
    Q = int(input[ptr])
    ptr += 1
    sh = int(input[ptr])
    ptr += 1
    sw = int(input[ptr])
    ptr += 1
    
    queries = []
    for _ in range(Q):
        d = input[ptr]
        ptr += 1
        a = int(input[ptr])
        ptr += 1
        queries.append((d, a))
    
    # Precompute C(h, w) for each cell (h, w)
    # C(h, w) = C(h + w - 2, h - 1) * C(H - h + W - w, H - h)
    total = 0
    C = [[0]*W for _ in range(H)]
    sum_total = 0
    for h in range(1, H+1):
        for w in range(1, W+1):
            a = h + w - 2
            b = h - 1
            c1 = comb(a, b)
            a2 = (H - h) + (W - w)
            b2 = (H - h)
            c2 = comb(a2, b2)
            c = c1 * c2 % MOD
            C_hw = c
            val = grid[h-1][w-1]
            sum_total = (sum_total + val * C_hw) % MOD
    
    # Initialize current position
    current_h = sh
    current_w = sw
    
    res = []
    for d, a in queries:
        # Compute C_hw for current cell after move
        if d == 'U':
            current_h -= 1
        elif d == 'D':
            current_h += 1
        elif d == 'L':
            current_w -= 1
        elif d == 'R':
            current_w += 1
        
        h = current_h
        w = current_w
        a_prev = grid[h-1][w-1]
        # Compute C(h, w)
        a_comb = h + w - 2
        b_comb = h - 1
        c1 = comb(a_comb, b_comb)
        a2_comb = (H - h) + (W - w)
        b2_comb = (H - h)
        c2 = comb(a2_comb, b2_comb)
        c_hw = c1 * c2 % MOD
        
        # Update sum_total
        delta = (a - a_prev) % MOD
        sum_total = (sum_total + delta * c_hw) % MOD
        grid[h-1][w-1] = a
        res.append(sum_total % MOD)
    
    print('
'.join(map(str, res)))

if __name__ == '__main__':
    main()