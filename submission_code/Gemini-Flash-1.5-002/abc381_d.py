def solve():
    n = int(input())
    a = list(map(int, input().split()))

    max_len = 0
    for i in range(n + 1):
        for j in range(i, n + 1):
            sub_array = a[i:j]
            if len(sub_array) % 2 == 0:
                is_1122 = True
                counts = {}
                for k in range(0, len(sub_array), 2):
                    if k + 1 >= len(sub_array) or sub_array[k] != sub_array[k+1]:
                        is_1122 = False
                        break
                    counts[sub_array[k]] = counts.get(sub_array[k], 0) + 2
                
                if is_1122:
                    all_twice = True
                    for count in counts.values():
                        if count != 2:
                            all_twice = False
                            break
                    if all_twice:
                        max_len = max(max_len, len(sub_array))

    print(max_len)

solve()