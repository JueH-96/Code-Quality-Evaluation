def solve():
    m = int(input())
    s1 = input()
    s2 = input()
    s3 = input()

    min_time = float('inf')
    found = False

    for digit in '0123456789':
        indices1 = [i for i, char in enumerate(s1) if char == digit]
        indices2 = [i for i, char in enumerate(s2) if char == digit]
        indices3 = [i for i, char in enumerate(s3) if char == digit]

        if not indices1 or not indices2 or not indices3:
            continue

        for idx1 in indices1:
            for idx2 in indices2:
                for idx3 in indices3:
                    for k1 in range(3):
                        t1 = idx1 + k1 * m
                        for k2 in range(3):
                            t2 = idx2 + k2 * m
                            for k3 in range(3):
                                t3 = idx3 + k3 * m
                                current_max_time = max(t1, t2, t3)
                                if s1[t1 % m] == digit and s2[t2 % m] == digit and s3[t3 % m] == digit:
                                    min_time = min(min_time, current_max_time)
                                    found = True

    if found:
        print(min_time)
    else:
        print(-1)

solve()