def solve():
    n_x = input().split()
    n = int(n_x[0])
    x = int(n_x[1])
    s_list_str = input().split()
    s_list = [int(s) for s in s_list_str]

    total_score = 0
    for score in s_list:
        if score <= x:
            total_score += score
    print(total_score)

solve()