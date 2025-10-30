# YOUR CODE HERE
def solve():
    n = int(input())
    a = list(map(int, input().split()))

    operations = []
    for i in range(n):
        if a[i] != i + 1:
            j = -1
            for k in range(i + 1, n):
                if a[k] == i + 1:
                    j = k
                    break
            if j != -1:
                for k in range(j, i, -1):
                    a[k], a[k-1] = a[k-1], a[k]
                    operations.append((k, k + 1))

    print(len(operations))
    for op in operations:
        print(op[0], op[1])

solve()