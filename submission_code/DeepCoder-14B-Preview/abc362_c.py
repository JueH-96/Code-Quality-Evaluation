def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    L = []
    R = []
    for _ in range(N):
        L.append(int(input[idx]))
        R.append(int(input[idx + 1]))
        idx += 2
    
    sum_L = sum(L)
    sum_R = sum(R)
    
    if sum_L > 0 or sum_R < 0:
        print("No")
        return
    
    x = L.copy()
    d = 0 - sum_L
    
    for i in range(N):
        if d == 0:
            break
        max_add = R[i] - x[i]
        add = min(max_add, d)
        x[i] += add
        d -= add
    
    print("Yes")
    print(' '.join(map(str, x)))

if __name__ == '__main__':
    main()