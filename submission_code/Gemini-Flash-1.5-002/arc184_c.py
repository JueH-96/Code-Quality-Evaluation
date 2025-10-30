def solve():
    n = int(input())
    a = list(map(int, input().split()))

    max_f = 0
    for i in range(1, 2**100 - a[-1]):
        f_i = 0
        for k in range(n):
            crease_index = i + a[k]
            
            level = 0
            temp_index = crease_index
            while temp_index > 0:
                level += 1
                temp_index //= 2

            if (level % 2) == 0:
                f_i += 1

        max_f = max(max_f, f_i)
    print(max_f)

solve()