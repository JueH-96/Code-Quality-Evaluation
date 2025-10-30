# YOUR CODE HERE
def solve():
    n = int(input())
    m = 0
    while (1 << m) < n:
        m += 1

    print(m)

    groups = []
    for i in range(m):
        group = []
        for j in range(1, n + 1):
            if (j >> i) & 1:
                group.append(j)
        groups.append(group)

    for group in groups:
        print(len(group), *group)

    s = input()
    bad_bottle = 0
    for i in range(m):
        if s[i] == '1':
            bad_bottle += (1 << i)
    print(bad_bottle)

solve()