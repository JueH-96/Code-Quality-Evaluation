# YOUR CODE HERE
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
    for i in range(1 << n):
        real_keys = []
        for j in range(n):
            if (i >> j) & 1:
                real_keys.append(j + 1)
        
        consistent = True
        for keys, result in tests:
            num_real_inserted = 0
            for key in keys:
                if key in real_keys:
                    num_real_inserted += 1
            
            if result == 'o':
                if num_real_inserted < k:
                    consistent = False
                    break
            else:
                if num_real_inserted >= k:
                    consistent = False
                    break
        
        if consistent:
            count += 1

    print(count)

solve()