def solve():
    s = input()
    t = input()
    n = len(s)
    x = []

    current_s = list(s)
    target_t = list(t)

    while current_s != target_t:
        best_next_s = None

        for i in range(n):
            if current_s[i] != target_t[i]:
                original_char = current_s[i]
                current_s[i] = target_t[i]
                next_s_str = "".join(current_s)

                if best_next_s is None or next_s_str < best_next_s:
                    best_next_s = next_s_str

                current_s[i] = original_char

        x.append(best_next_s)
        current_s = list(best_next_s)

    print(len(x))
    for val in x:
        print(val)

solve()