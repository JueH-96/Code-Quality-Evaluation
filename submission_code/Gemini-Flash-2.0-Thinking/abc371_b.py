def solve():
    n, m = map(int, input().split())
    babies = []
    for _ in range(m):
        a, b = input().split()
        babies.append((int(a), b))

    taro_assigned = {}

    for i in range(m):
        family_id, gender = babies[i]
        is_male = gender == 'M'
        is_taro = False

        if is_male:
            if family_id not in taro_assigned:
                is_taro = True
                taro_assigned[family_id] = True

        if is_taro:
            print("Yes")
        else:
            print("No")

solve()