def solve():
    s = input()
    n = int(input())
    q_indices = [i for i, char in enumerate(s) if char == '?']
    num_q = len(q_indices)
    max_val = -1

    def generate(index, current_s):
        nonlocal max_val
        if index == num_q:
            try:
                val = int("".join(current_s), 2)
                if val <= n:
                    max_val = max(max_val, val)
            except ValueError:
                pass
            return

        q_index = q_indices[index]

        # Try replacing with 0
        temp_s_0 = list(current_s)
        temp_s_0[q_index] = '0'
        generate(index + 1, temp_s_0)

        # Try replacing with 1
        temp_s_1 = list(current_s)
        temp_s_1[q_index] = '1'
        generate(index + 1, temp_s_1)

    generate(0, list(s))
    print(max_val)

solve()