def solve():
    s = input()
    t = input()
    n = len(s)
    m = len(t)
    if n != m:
        print("No")
        return

    count_s = {}
    count_t = {}
    at_s = 0
    at_t = 0

    for char in s:
        if char == '@':
            at_s += 1
        else:
            count_s[char] = count_s.get(char, 0) + 1

    for char in t:
        if char == '@':
            at_t += 1
        else:
            count_t[char] = count_t.get(char, 0) + 1

    allowed_chars = set('atcoder')

    for char_code in range(ord('a'), ord('z') + 1):
        char = chr(char_code)
        count_s_char = count_s.get(char, 0)
        count_t_char = count_t.get(char, 0)

        if char not in allowed_chars:
            if count_s_char != count_t_char:
                print("No")
                return

    for char in allowed_chars:
        count_s_char = count_s.get(char, 0)
        count_t_char = count_t.get(char, 0)

        if count_s_char > count_t_char:
            needed = count_s_char - count_t_char
            at_t -= needed
            if at_t < 0:
                print("No")
                return
        elif count_t_char > count_s_char:
            needed = count_t_char - count_s_char
            at_s -= needed
            if at_s < 0:
                print("No")
                return

    print("Yes")

solve()