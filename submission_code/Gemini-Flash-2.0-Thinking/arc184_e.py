def solve():
    n, m = map(int, input().split())
    sequences = []
    for _ in range(n):
        sequences.append(list(map(int, input().split())))

    mod = 998244353

    def apply_operation(arr):
        m_len = len(arr)
        new_arr = [0] * m_len
        current_sum = 0
        for k in range(m_len):
            current_sum = (current_sum + arr[k]) % 2
            new_arr[k] = current_sum
        return new_arr

    def calculate_f(arr1, arr2):
        if arr1 == arr2:
            return 0

        temp_arr1 = list(arr1)
        temp_arr2 = list(arr2)

        for x in range(1, m * 4 + 5):
            temp_arr1 = apply_operation(temp_arr1)
            temp_arr2 = apply_operation(temp_arr2)
            if temp_arr1 == temp_arr2:
                return x
        return 0

    total_f = 0
    for i in range(n):
        for j in range(i, n):
            f_val = calculate_f(sequences[i], sequences[j])
            total_f = (total_f + f_val) % mod

    print(total_f)

solve()