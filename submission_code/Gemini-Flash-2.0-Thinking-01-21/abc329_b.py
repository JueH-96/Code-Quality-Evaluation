def solve():
    n = int(input())
    a = list(map(int, input().split()))

    max_val = max(a)
    not_largest_list = []
    for x in a:
        if x != max_val:
            not_largest_list.append(x)

    print(max(not_largest_list))

solve()