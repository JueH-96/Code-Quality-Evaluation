def solve():
    s = input()
    n = len(s)
    mod = 998244353

    def is_ddos(sub):
        if len(sub) != 4:
            return False
        if not ('A' <= sub[0] <= 'Z' and 'A' <= sub[1] <= 'Z' and 'a' <= sub[2] <= 'z' and 'A' <= sub[3] <= 'Z'):
            return False
        return sub[0] == sub[1]

    def count_non_ddos(current_s):
        for i in range(len(current_s) - 3):
            sub = current_s[i:i+4]
            if is_ddos(sub):
                return 0
        return 1

    q_indices = [i for i, char in enumerate(s) if char == '?']
    num_q = len(q_indices)
    ans = 0

    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

    def generate(index, current_string):
        nonlocal ans
        if index == num_q:
            if not contains_ddos_subsequence(current_string):
                ans = (ans + 1) % mod
            return

        q_index = q_indices[index]
        original_char = s[q_index]

        for char in chars:
            temp_s = list(current_string)
            temp_s[q_index] = char
            generate(index + 1, "".join(temp_s))

    def contains_ddos_subsequence(text):
        n = len(text)
        for i in range(n):
            if 'A' <= text[i] <= 'Z':
                for j in range(i + 1, n):
                    if 'A' <= text[j] <= 'Z' and text[i] == text[j]:
                        for k in range(j + 1, n):
                            if 'a' <= text[k] <= 'z':
                                for l in range(k + 1, n):
                                    if 'A' <= text[l] <= 'Z':
                                        return True
        return False

    dp = {}

    def count_non_ddos_dp(index, current_subsequence_state):
        if index == n:
            return 1

        if (index, current_subsequence_state) in dp:
            return dp[(index, current_subsequence_state)]

        count = 0
        possible_chars = []
        if s[index] == '?':
            possible_chars.extend([chr(ord('A') + i) for i in range(26)])
            possible_chars.extend([chr(ord('a') + i) for i in range(26)])
        else:
            possible_chars.append(s[index])

        for char in possible_chars:
            new_state = list(current_subsequence_state)
            new_state.append(char)

            is_ddos_found = False
            for i_sub in range(len(new_state) - 3):
                sub = new_state[i_sub:i_sub+4]
                if len(sub) == 4 and 'A' <= sub[0] <= 'Z' and 'A' <= sub[1] <= 'Z' and 'a' <= sub[2] <= 'z' and 'A' <= sub[3] <= 'Z' and sub[0] == sub[1]:
                    is_ddos_found = True
                    break
                if is_ddos_found:
                    break

            if not is_ddos_found:
                count = (count + count_non_ddos_dp(index + 1, tuple(new_state))) % mod

        dp[(index, current_subsequence_state)] = count
        return count

    q_count = s.count('?')
    total_strings = pow(52, q_count, mod)

    def check(generated_s):
        return not contains_ddos_subsequence(generated_s)

    from itertools import product

    ans = 0
    q_indices = [i for i, char in enumerate(s) if char == '?']
    n_q = len(q_indices)
    chars_upper = [chr(ord('A') + i) for i in range(26)]
    chars_lower = [chr(ord('a') + i) for i in range(26)]
    all_chars = chars_upper + chars_lower

    for replacements in product(all_chars, repeat=n_q):
        temp_s_list = list(s)
        for i in range(n_q):
            temp_s_list[q_indices[i]] = replacements[i]
        generated_string = "".join(temp_s_list)
        if not contains_ddos_subsequence(generated_string):
            ans = (ans + 1) % mod

    print(ans)

solve()