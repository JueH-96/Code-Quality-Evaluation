def solve():
    n = int(input())
    s = input()
    max_len = 0
    for mid_index in range(n):
        if s[mid_index] == '/':
            max_len = max(max_len, 1)  # Single '/' is an 11/22 string

            for k in range(1, n // 2 + 1):
                length = 2 * k + 1
                start_index = mid_index - k
                end_index = mid_index + k

                if start_index >= 0 and end_index < n:
                    is_1122_substring = True
                    # Check '1's before '/'
                    for i in range(start_index, mid_index):
                        if s[i] != '1':
                            is_1122_substring = False
                            break
                    if not is_1122_substring:
                        continue

                    # Check '2's after '/'
                    for i in range(mid_index + 1, end_index + 1):
                        if s[i] != '2':
                            is_1122_substring = False
                            break

                    if is_1122_substring:
                        max_len = max(max_len, length)
    print(max_len)