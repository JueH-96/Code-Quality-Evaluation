def solve():
    n, m, k = map(int, input().split())
    tests = []
    for _ in range(m):
        line = input().split()
        c = int(line[0])
        keys = [int(x) for x in line[1:c+1]]
        r = line[-1]
        tests.append((keys, r))

    valid_combinations_count = 0
    for i in range(2**n):
        real_keys = set()
        for j in range(n):
            if (i >> j) & 1:
                real_keys.add(j + 1)

        is_consistent = True
        for keys_inserted, result in tests:
            num_real_inserted = 0
            for key in keys_inserted:
                if key in real_keys:
                    num_real_inserted += 1

            if result == 'o':
                if num_real_inserted < k:
                    is_consistent = False
                    break
            elif result == 'x':
                if num_real_inserted >= k:
                    is_consistent = False
                    break

        if is_consistent:
            valid_combinations_count += 1

    print(valid_combinations_count)

solve()