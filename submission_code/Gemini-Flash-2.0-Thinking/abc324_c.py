def solve():
    n, t_prime = input().split()
    n = int(n)
    s_list = [input() for _ in range(n)]

    result_indices = []

    for idx, s in enumerate(s_list):
        is_possible = False

        # Condition 1: Equal
        if s == t_prime:
            is_possible = True
        # Condition 2: Insertion in s
        elif len(t_prime) == len(s) + 1:
            for i in range(len(t_prime)):
                temp_s_prime = list(t_prime)
                inserted_char = temp_s_prime[i]
                temp_s = list(s)
                for j in range(len(s) + 1):
                    temp_s_insert = list(s)
                    temp_s_insert.insert(j, inserted_char)
                    if "".join(temp_s_insert) == t_prime:
                        is_possible = True
                        break
                if is_possible:
                    break

        elif len(t_prime) == len(s) + 1:
            for i in range(len(s) + 1):
                for char_code in range(ord('a'), ord('z') + 1):
                    inserted_char = chr(char_code)
                    temp_s = list(s)
                    temp_s.insert(i, inserted_char)
                    if "".join(temp_s) == t_prime:
                        is_possible = True
                        break
                if is_possible:
                    break

        # Condition 3: Deletion from s
        elif len(t_prime) == len(s) - 1:
            for i in range(len(s)):
                temp_s = list(s)
                deleted_char = temp_s.pop(i)
                if "".join(temp_s) == t_prime:
                    is_possible = True
                    break

        # Condition 4: Change in s
        elif len(t_prime) == len(s):
            for i in range(len(s)):
                original_char = s[i]
                for char_code in range(ord('a'), ord('z') + 1):
                    new_char = chr(char_code)
                    if new_char != original_char:
                        temp_s = list(s)
                        temp_s[i] = new_char
                        if "".join(temp_s) == t_prime:
                            is_possible = True
                            break
                if is_possible:
                    break

        if is_possible:
            result_indices.append(idx + 1)

    print(len(result_indices))
    print(*result_indices)

solve()