def solve():
    n, m, k = map(int, input().split())
    tests = []
    for _ in range(m):
        line = input().split()
        c = int(line[0])
        keys = list(map(int, line[1:c+1]))
        result = line[-1]
        tests.append((keys, result))

    count = 0
    for i in range(2**n):
        real_keys = set()
        for j in range(n):
            if (i >> j) & 1:
                real_keys.add(j + 1)
        
        valid = True
        for keys, result in tests:
            real_count = 0
            for key in keys:
                if key in real_keys:
                    real_count += 1
            
            if result == 'o':
                if real_count < k:
                    valid = False
                    break
            else:
                if real_count >= k:
                    valid = False
                    break
        
        if valid:
            count += 1
    
    print(count)

solve()