import sys

def main():
    H, W = map(int, sys.stdin.readline().split())
    rows = []
    for _ in range(H):
        s = sys.stdin.readline().strip()
        bitmask = int(s, 2)
        rows.append(bitmask)
    
    max_mask = 1 << W
    cnt = [0] * max_mask
    for row in rows:
        cnt[row] += 1
    
    # Precompute f[x] = min(popcount(x), W - popcount(x))
    f = [0] * max_mask
    for x in range(max_mask):
        pop = bin(x).count('1')
        f[x] = min(pop, W - pop)
    
    # Define FWHT function
    def fwht(a):
        n = len(a)
        h = 1
        while h < n:
            for i in range(0, n, h * 2):
                for j in range(i, i + h):
                    x = a[j]
                    y = a[j + h]
                    a[j] = x + y
                    a[j + h] = x - y
            h *= 2
    
    # Copy and apply FWHT on cnt and f
    a_fwht = cnt.copy()
    fwht(a_fwht)
    
    b_fwht = f.copy()
    fwht(b_fwht)
    
    # Element-wise multiplication
    for i in range(max_mask):
        a_fwht[i] *= b_fwht[i]
    
    # Apply inverse FWHT (another FWHT and divide by n)
    fwht(a_fwht)
    
    n = max_mask
    conv = [a_fwht[i] // n for i in range(max_mask)]
    
    print(min(conv))
    
if __name__ == "__main__":
    main()