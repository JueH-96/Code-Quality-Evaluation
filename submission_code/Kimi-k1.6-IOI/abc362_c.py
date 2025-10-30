def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    pairs = []
    sum_L = 0
    sum_R = 0
    for _ in range(N):
        l = int(input[idx])
        r = int(input[idx+1])
        pairs.append((l, r))
        sum_L += l
        sum_R += r
        idx += 2
    
    if sum_L > 0 or sum_R < 0:
        print("No")
        return
    
    needed = -sum_L
    X = [l for l, r in pairs]
    for i in range(N):
        l, r = pairs[i]
        max_add = r - X[i]
        add = min(needed, max_add)
        X[i] += add
        needed -= add
        if needed == 0:
            break
    
    print("Yes")
    print(' '.join(map(str, X)))

if __name__ == "__main__":
    main()