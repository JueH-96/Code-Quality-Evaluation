import sys

def fwht(a, inverse=False):
    n = len(a)
    h = 1
    while h < n:
        for i in range(0, n, 2 * h):
            for j in range(i, i + h):
                x = a[j]
                y = a[j + h]
                a[j] = x + y
                a[j + h] = x - y
        h <<= 1
    if inverse:
        n_val = n
        for i in range(n):
            a[i] //= n_val

def main():
    data = sys.stdin.read().splitlines()
    if not data:
        return
    H, W = map(int, data[0].split())
    grid = data[1:1 + H]
    
    n = 1 << W
    F_arr = [0] * n
    for i in range(H):
        s = grid[i].strip()
        num = 0
        for char in s:
            num = (num << 1) | (1 if char == '1' else 0)
        F_arr[num] += 1
        
    G_arr = [0] * n
    for mask in range(n):
        cnt = bin(mask).count('1')
        G_arr[mask] = min(cnt, W - cnt)
        
    fwht(F_arr)
    fwht(G_arr)
    
    H_arr = [F_arr[i] * G_arr[i] for i in range(n)]
    
    fwht(H_arr, inverse=True)
    
    ans = min(H_arr)
    print(ans)

if __name__ == '__main__':
    main()