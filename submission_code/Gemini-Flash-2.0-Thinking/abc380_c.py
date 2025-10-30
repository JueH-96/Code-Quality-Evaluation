def solve():
    n, k = map(int, input().split())
    s = input()
    k -= 1  # 0-based index

    blocks = []
    start = -1
    for i in range(n):
        if s[i] == '1':
            if start == -1:
                start = i
        else:
            if start != -1:
                blocks.append((start, i - 1))
                start = -1
    if start != -1:
        blocks.append((start, n - 1))

    start_k_minus_1, end_k_minus_1 = blocks[k - 1]
    start_k, end_k = blocks[k]

    moved_block = s[start_k : end_k + 1]

    temp_s_list = list(s)
    moved_block_list = temp_s_list[start_k : end_k + 1]
    del temp_s_list[start_k : end_k + 1]

    insertion_point = end_k_minus_1 + 1
    temp_s_list[insertion_point:insertion_point] = moved_block_list

    print("".join(temp_s_list))

solve()