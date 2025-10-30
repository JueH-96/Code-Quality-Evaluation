import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    P = list(map(int, data[1:1+n]))
    
    size = 1
    while size < n:
        size *= 2
        
    seg = [0] * (2 * size + 1)
    
    for i in range(n):
        seg[size + i] = 1
    for i in range(n, size):
        seg[size + i] = 0
        
    for i in range(size - 1, 0, -1):
        seg[i] = seg[2*i] + seg[2*i+1]
        
    ans = [0] * n
    
    for i in range(n-1, -1, -1):
        k = P[i]
        idx = 1
        while idx < size:
            if k <= seg[2*idx]:
                idx = 2*idx
            else:
                k -= seg[2*idx]
                idx = 2*idx + 1
                
        pos = idx - size
        ans[pos] = i + 1
        
        idx = size + pos
        seg[idx] = 0
        while idx > 1:
            idx //= 2
            seg[idx] = seg[2*idx] + seg[2*idx+1]
            
    print(" ".join(map(str, ans)))
    
if __name__ == "__main__":
    main()