def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    n = int(data[0])
    x = int(data[1])
    A = list(map(int, data[2:2 + n - 1]))
    
    total_so_far = sum(A)
    low = min(A)
    high = max(A)
    
    for k in range(0, 101):
        total_all = total_so_far + k
        m_val = min(low, k)
        M_val = max(high, k)
        if total_all - m_val - M_val >= x:
            print(k)
            return
    
    print(-1)

if __name__ == '__main__':
    main()