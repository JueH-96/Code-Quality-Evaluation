def solve():
    n = int(input())
    a = list(map(int, input().split()))

    max_len = 0
    for i in range(n):
        for j in range(i, n):
            sub_array = a[i:j+1]
            len_sub = len(sub_array)

            if len_sub % 2 != 0:
                continue

            is_1122 = True
            
            counts = {}
            for x in sub_array:
                counts[x] = counts.get(x, 0) + 1
            
            for x in counts:
                if counts[x] != 2:
                    is_1122 = False
                    break
            
            if not is_1122:
                continue

            for k in range(0, len_sub, 2):
                if k + 1 >= len_sub or sub_array[k] != sub_array[k+1]:
                    is_1122 = False
                    break

            if is_1122:
                max_len = max(max_len, len_sub)

    print(max_len)

solve()