def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    N = int(data[index])
    index += 1
    M = int(data[index])
    index += 1
    K = int(data[index])
    index += 1
    
    tests = []
    
    for _ in range(M):
        C_i = int(data[index])
        index += 1
        keys = []
        for _ in range(C_i):
            keys.append(int(data[index]) - 1)
            index += 1
        result = data[index]
        index += 1
        tests.append((keys, result))
    
    valid_combinations = 0
    
    # Iterate over all possible combinations of keys being real or dummy
    for mask in range(1 << N):
        valid = True
        
        for keys, result in tests:
            count_real = sum((mask & (1 << k)) > 0 for k in keys)
            
            if result == 'o' and count_real < K:
                valid = False
                break
            elif result == 'x' and count_real >= K:
                valid = False
                break
        
        if valid:
            valid_combinations += 1
    
    print(valid_combinations)

if __name__ == "__main__":
    main()