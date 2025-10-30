def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    S = input[idx]
    idx += 1
    C = list(map(int, input[idx:idx+N]))
    
    prefix0 = [0] * (N + 1)
    prefix1 = [0] * (N + 1)
    
    for j in range(1, N + 1):
        current = int(S[j-1])
        # Compute for prefix0
        if j % 2 == 1:
            d0 = 0
        else:
            d0 = 1
        cost0 = 0 if current == d0 else C[j-1]
        prefix0[j] = prefix0[j-1] + cost0
        
        # Compute for prefix1
        if j % 2 == 1:
            d1 = 1
        else:
            d1 = 0
        cost1 = 0 if current == d1 else C[j-1]
        prefix1[j] = prefix1[j-1] + cost1
    
    suffix0 = [0] * (N + 2)
    suffix1 = [0] * (N + 2)
    
    for j in range(N, 0, -1):
        current = int(S[j-1])
        cost0 = 0 if current == 0 else C[j-1]
        cost1 = 0 if current == 1 else C[j-1]
        suffix0[j] = cost0 + suffix1[j+1]
        suffix1[j] = cost1 + suffix0[j+1]
    
    min_total = float('inf')
    
    for i in range(1, N):
        for s in [0, 1]:
            if i % 2 == 1:
                val = s
            else:
                val = 1 - s
            if val == 0:
                sc = suffix0[i+1]
            else:
                sc = suffix1[i+1]
            if s == 0:
                pc = prefix0[i]
            else:
                pc = prefix1[i]
            total = pc + sc
            if total < min_total:
                min_total = total
    
    print(min_total)

if __name__ == "__main__":
    main()