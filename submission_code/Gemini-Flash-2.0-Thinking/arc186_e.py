def solve():
    n, m, k = map(int, input().split())
    x = list(map(int, input().split()))
    mod = 998244353

    def is_subsequence(sub, seq):
        i = 0
        j = 0
        while i < len(sub) and j < len(seq):
            if sub[i] == seq[j]:
                i += 1
            j += 1
        return i == len(sub)

    def count_sequences(target_forbidden):
        count = 0
        for i in range(k**n):
            seq = []
            temp = i
            for _ in range(n):
                seq.append(temp % k + 1)
                temp //= k
            seq.reverse()

            forbidden_found = False
            for f in target_forbidden:
                if not is_subsequence(f, seq):
                    forbidden_found = True
                    break

            if forbidden_found:
                count += 1
        return count

    def get_all_sequences(length, alphabet_size):
        if length == 0:
            yield []
        else:
            for seq in get_all_sequences(length - 1, alphabet_size):
                for i in range(1, alphabet_size + 1):
                    yield seq + [i]

    def check(a, forbidden_seq):
        non_subsequences = []
        for seq_m in get_all_sequences(m, k):
            if not is_subsequence(seq_m, a):
                non_subsequences.append(tuple(seq_m))

        return set(non_subsequences) == {tuple(forbidden_seq)}

    ans = 0
    for seq_n_int in range(k**n):
        a = []
        temp = seq_n_int
        for _ in range(n):
            a.append(temp % k + 1)
            temp //= k
        a.reverse()

        if check(a, x):
            ans = (ans + 1) % mod

    print(ans)

solve()