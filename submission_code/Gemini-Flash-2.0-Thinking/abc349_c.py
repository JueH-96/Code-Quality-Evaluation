def solve():
    s = input()
    t = input()
    n = len(s)

    # Check condition 1
    if len(t) == 3:
        t_ptr = 0
        for char_s in s:
            if t_ptr < 3 and char_s.upper() == t[t_ptr]:
                t_ptr += 1
        if t_ptr == 3:
            print("Yes")
            return

    # Check condition 2
    if len(t) == 3 and t[2] == 'X':
        t_ptr = 0
        t_prefix = t[:2]
        for char_s in s:
            if t_ptr < 2 and char_s.upper() == t_prefix[t_ptr]:
                t_ptr += 1
        if t_ptr == 2:
            print("Yes")
            return

    print("No")

solve()