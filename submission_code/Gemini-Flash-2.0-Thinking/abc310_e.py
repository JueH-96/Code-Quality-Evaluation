def solve():
    n = int(input())
    s = input()
    a = [int(c) for c in s]

    def nand(x, y):
        return 1 if x == 0 or y == 0 else 0

    total_sum = 0
    for i in range(n):
        current_f = -1
        for j in range(i, n):
            if j == i:
                current_f = a[i]
            else:
                current_f = nand(current_f, a[j])
            total_sum += current_f
    print(total_sum)

solve()