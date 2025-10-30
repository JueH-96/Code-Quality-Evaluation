def main():
    import sys
    data = sys.stdin.read().split()
    A = int(data[0])
    B = int(data[1])
    M = int(data[2])
    
    if A == 2 and B == 2:
        print(0)
        return
        
    n = A * B - 1
    fact = 1
    for i in range(1, n + 1):
        fact = (fact * i) % M
        
    denom = 1
    for i in range(B):
        if i < B - 1:
            row_len = A
        else:
            row_len = A - 1
            
        for j in range(row_len):
            if j < A - 1:
                conj = B
            else:
                conj = B - 1
            hook_val = row_len - j + conj - i - 1
            denom = (denom * hook_val) % M
            
    denom_inv = pow(denom, M - 2, M)
    syt_count = fact * denom_inv % M
    ans = (A - 1) * syt_count % M
    print(ans)

if __name__ == '__main__':
    main()