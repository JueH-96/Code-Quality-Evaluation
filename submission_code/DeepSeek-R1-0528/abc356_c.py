def main():
    import sys
    data = sys.stdin.read().splitlines()
    if not data: 
        print(0)
        return
        
    first_line = data[0].split()
    N = int(first_line[0])
    M = int(first_line[1])
    K = int(first_line[2])
    
    tests = []
    for i in range(1, 1+M):
        parts = data[i].split()
        c = int(parts[0])
        keys = list(map(int, parts[1:1+c]))
        res_char = parts[1+c]
        mask = 0
        for key in keys:
            mask |= (1 << (key-1))
        tests.append((mask, res_char))
    
    max_state = 1 << N
    popcount_arr = [0] * max_state
    for i in range(max_state):
        popcount_arr[i] = bin(i).count('1')
    
    count_valid = 0
    for state in range(max_state):
        valid = True
        for (mask, res) in tests:
            common = state & mask
            cnt = popcount_arr[common]
            if (res == 'o' and cnt < K) or (res == 'x' and cnt >= K):
                valid = False
                break
        if valid:
            count_valid += 1
            
    print(count_valid)

if __name__ == "__main__":
    main()