def solve():
    n, q = map(int, input().split())
    s = input()

    def is_1122_string(t):
        lt = len(t)
        if lt % 2 == 0:
            return False
        mid = (lt + 1) // 2
        for i in range(mid - 1):
            if i >= lt or t[i] != '1':
                return False
        if mid - 1 >= lt or t[mid - 1] != '/':
            return False
        for i in range(mid, lt):
            if i >= lt or t[i] != '2':
                return False
        return True

    def find_max_1122_subsequence(text):
        n_text = len(text)
        max_len = 0
        for i in range(1 << n_text):
            subsequence = ""
            for j in range(n_text):
                if (i >> j) & 1:
                    subsequence += text[j]
            if is_1122_string(subsequence):
                max_len = max(max_len, len(subsequence))
        return max_len

    for _ in range(q):
        l, r = map(int, input().split())
        sub = s[l-1:r]

        n_sub = len(sub)
        max_len = 0

        for length in range(1, n_sub + 1, 2):
            num_ones = (length - 1) // 2
            num_twos = (length - 1) // 2

            for i_slash in range(n_sub):
                if sub[i_slash] == '/':
                    count_ones_before = 0
                    ones_indices = []
                    for k in range(i_slash):
                        if sub[k] == '1':
                            ones_indices.append(k)

                    if len(ones_indices) >= num_ones:
                        count_twos_after = 0
                        twos_indices = []
                        for k in range(i_slash + 1, n_sub):
                            if sub[k] == '2':
                                twos_indices.append(k)

                        if len(twos_indices) >= num_twos:
                            # Check if a valid subsequence can be formed
                            import itertools

                            for ones_comb in itertools.combinations(range(i_slash), num_ones):
                                for twos_comb in itertools.combinations(range(i_slash + 1, n_sub), num_twos):
                                    combined_indices = sorted(list(ones_comb) + [i_slash] + list(twos_comb))
                                    potential_subsequence = ""
                                    temp_indices = sorted(list(ones_comb))

                                    is_possible = True
                                    for idx in temp_indices:
                                        if idx >= i_slash:
                                            is_possible = False
                                            break
                                        potential_subsequence += sub[idx]
                                    if not is_possible:
                                        continue

                                    potential_subsequence += '/'

                                    temp_indices_two = sorted(list(twos_comb))
                                    for idx in temp_indices_two:
                                        if idx <= i_slash:
                                            is_possible = False
                                            break
                                        potential_subsequence += sub[idx]
                                    if is_possible and is_1122_string(potential_subsequence):
                                        max_len = max(max_len, length)
                                        break
                                if max_len == length:
                                    break
                            if max_len == length:
                                break
                if max_len == length:
                    break

        print(max_len)

solve()