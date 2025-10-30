def solve():
    S = input()
    X = input()
    Y = input()

    count_s_x = X.count('0')
    count_s_y = Y.count('0')
    count_t_x = len(X) - count_s_x
    count_t_y = len(Y) - count_s_y

    if count_s_x == count_s_y:
        print("Yes")
        return

    len_s = len(S)

    if count_s_x > count_s_y:
        diff = count_s_x - count_s_y
        len_f_x = len_s * count_s_x
        len_f_y = len_s * count_s_y
    else:
        diff = count_s_y - count_s_x
        len_f_x = len_s * count_s_x
        len_f_y = len_s * count_s_y

    if count_s_x == 0 and count_s_y == 0:
        print("Yes")
        return

    if S == "":
        print("Yes")
        return

    print("No")

t = int(input())
for _ in range(t):
    solve()