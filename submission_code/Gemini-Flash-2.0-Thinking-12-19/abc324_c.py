def solve():
    n, t_prime = input().split()
    n = int(n)
    s_list = [input() for _ in range(n)]

    def is_equal(s1, s2):
        return s1 == s2

    def is_insertion(s1, s2):
        if len(s2) != len(s1) + 1:
            return False
        i = 0
        j = 0
        diff_count = 0
        while i < len(s1) and j < len(s2):
            if s1[i] == s2[j]:
                i += 1
                j += 1
            else:
                diff_count += 1
                j += 1
        return diff_count <= 1 and j == len(s2) and i == len(s1)

    def is_deletion(s1, s2):
        if len(s2) != len(s1) - 1:
            return False
        i = 0
        j = 0
        diff_count = 0
        while i < len(s1) and j < len(s2):
            if s1[i] == s2[j]:
                i += 1
                j += 1
            else:
                diff_count += 1
                i += 1
        return diff_count <= 1 and i == len(s1) and j == len(s2)

    def is_change(s1, s2):
        if len(s2) != len(s1):
            return False
        diff_count = 0
        for i in range(len(s1)):
            if s1[i] != s2[i]:
                diff_count += 1
        return diff_count == 1

    result_indices = []
    for i in range(n):
        s_i = s_list[i]
        if is_equal(s_i, t_prime) or is_insertion(s_i, t_prime) or is_deletion(s_i, t_prime) or is_change(s_i, t_prime):
            result_indices.append(i + 1)

    print(len(result_indices))
    print(*(result_indices))

solve()