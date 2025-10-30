def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    N = int(data[index])
    index += 1
    
    qr = []
    for _ in range(N):
        q = int(data[index])
        r = int(data[index + 1])
        qr.append((q, r))
        index += 2
    
    Q = int(data[index])
    index += 1
    
    results = []
    for _ in range(Q):
        t = int(data[index])
        d = int(data[index + 1])
        index += 2
        
        q_val, r_val = qr[t - 1]
        current = d % q_val
        delta = (r_val - current) % q_val
        x = d + delta
        results.append(str(x))
    
    print('
'.join(results))

if __name__ == '__main__':
    main()