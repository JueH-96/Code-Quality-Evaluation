def solve():
    n, k = map(int, input().split())
    s = input()
    q_indices = [i for i, char in enumerate(s) if char == '?']
    num_q = len(q_indices)
    ans = 0
    mod = 998244353

    def is_good(sub):
        counts = {'A': 0, 'B': 0, 'C': 0}
        for char in sub:
            counts[char] += 1
        ca, cb, cc = counts['A'], counts['B'], counts['C']
        return (ca - cb) % 2 == 0 and (cb - cc) % 2 == 0

    for i in range(3**num_q):
        temp = i
        replaced_s_list = list(s)
        for j in range(num_q):
            digit = temp % 3
            temp //= 3
            if digit == 0:
                replaced_s_list[q_indices[j]] = 'A'
            elif digit == 1:
                replaced_s_list[q_indices[j]] = 'B'
            else:
                replaced_s_list[q_indices[j]] = 'C'
        replaced_s = "".join(replaced_s_list)

        good_substring_count = 0
        for length in range(1, n + 1):
            for start in range(n - length + 1):
                substring = replaced_s[start : start + length]
                if is_good(substring):
                    good_substring_count += 1

        if good_substring_count >= k:
            ans = (ans + 1) % mod

    print(ans)

solve()