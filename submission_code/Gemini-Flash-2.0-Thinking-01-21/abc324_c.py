def solve():
    n, t_prime = input().split()
    n = int(n)
    s_list = [input() for _ in range(n)]

    def is_equal(s1, s2):
        return s1 == s2

    def is_one_insert(s1, s2):
        if len(s2) != len(s1) + 1:
            return False
        for i in range(len(s2) + 1):
            temp_s1_list = list(s1)
            for char_code in range(ord('a'), ord('z') + 1):
                char_to_insert = chr(char_code)
                temp_s1_list_copy = list(temp_s1_list)
                temp_s1_list_copy.insert(i, char_to_insert)
                if "".join(temp_s1_list_copy) == s2:
                    return True
        return False

    def is_one_delete(s1, s2):
        if len(s2) != len(s1) - 1:
            return False
        for i in range(len(s1)):
            temp_s1_list = list(s1)
            del temp_s1_list[i]
            if "".join(temp_s1_list) == s2:
                return True
        return False

    def is_one_change(s1, s2):
        if len(s2) != len(s1):
            return False
        diff_count = 0
        for i in range(len(s1)):
            if s1[i] != s2[i]:
                diff_count += 1
        return diff_count == 1

    result_indices = []
    for i in range(n):
        s = s_list[i]
        if is_equal(s, t_prime):
            result_indices.append(i + 1)
        elif is_one_insert(s, t_prime):
            result_indices.append(i + 1)
        elif is_one_delete(s, t_prime):
            result_indices.append(i + 1)
        elif is_one_change(s, t_prime):
            result_indices.append(i + 1)

    print(len(result_indices))
    print(*(result_indices))

if __name__ == "__main__":
    solve()