def solve():
    n = int(input())
    a = list(map(int, input().split()))

    operations = []
    a_copy = list(a)

    for i in range(1, n + 1):
        if a_copy[i - 1] != i:
            j = -1
            for k in range(i, n):
                if a_copy[k] == i:
                    j = k + 1
                    break
            if j != -1:
                a_copy[i - 1], a_copy[j - 1] = a_copy[j - 1], a_copy[i - 1]
                operations.append((i, j))

    print(len(operations))
    for op in operations:
        print(op[0], op[1])

solve()