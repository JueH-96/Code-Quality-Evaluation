import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    try:
        N = int(data[0])
    except Exception:
        return
    
    if N == 1:
        print(0)
        return
        
    M = N - 1
    d = 1
    current = 0
    while True:
        if d == 1:
            count_d = 9
        else:
            base = (d + 1) // 2
            count_d = 9 * (10 ** (base - 1))
            
        if current + count_d >= M:
            break
            
        current += count_d
        d += 1
        
    offset = M - current
    left_size = (d + 1) // 2
    start = 10 ** (left_size - 1)
    left_num = start + offset - 1
    s = str(left_num)
    if d % 2 == 0:
        res = s + s[::-1]
    else:
        res = s + s[:-1][::-1]
    print(res)

if __name__ == '__main__':
    main()