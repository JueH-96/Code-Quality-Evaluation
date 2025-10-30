def solve():
    n_l = input().split()
    n = int(n_l[0])
    l = int(n_l[1])
    a = list(map(int, input().split()))

    passed_count = 0
    for score in a:
        if score >= l:
            passed_count += 1
    print(passed_count)

solve()