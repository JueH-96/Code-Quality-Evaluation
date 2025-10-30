def main():
    import sys
    data = iter(sys.stdin.read().strip().split())
    N = int(next(data))
    M = int(next(data))
    K = int(next(data))
    
    setMasks = []
    isOpen = []
    for _ in range(M):
        C = int(next(data))
        mask = 0
        for _ in range(C):
            key = int(next(data))
            mask |= (1 << (key - 1))
        result = next(data)
        setMasks.append(mask)
        isOpen.append(result == 'o')
    
    answer = 0
    # Enumerate all subsets of keys
    for s in range(1 << N):
        valid = True
        for i in range(M):
            # Count how many of the inserted keys are "real" under subset s
            count = (s & setMasks[i]).bit_count()
            if isOpen[i]:
                # Door opened => need at least K real keys
                if count < K:
                    valid = False
                    break
            else:
                # Door did not open => fewer than K real keys
                if count >= K:
                    valid = False
                    break
        if valid:
            answer += 1
    
    print(answer)

# Do not forget to call main()
main()