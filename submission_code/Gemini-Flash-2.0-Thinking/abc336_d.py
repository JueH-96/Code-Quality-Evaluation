def solve():
    n = int(input())
    a = list(map(int, input().split()))

    def can_form(k):
        length = 2 * k - 1
        if length > n:
            return False

        for i in range(n - length + 1):
            sub_array = a[i : i + length]
            peak_index_sub = k - 1
            peak_value = k

            if sub_array[peak_index_sub] < peak_value:
                continue

            possible = True
            for j in range(k - 1):
                if sub_array[peak_index_sub - 1 - j] < peak_value - 1 - j:
                    possible = False
                    break
                if sub_array[peak_index_sub + 1 + j] < peak_value - 1 - j:
                    possible = False
                    break
            if possible:
                return True
        return False

    max_k = 0
    for k in range(1, (n + 1) // 2 + 2):
        if can_form(k):
            max_k = k
        else:
            break
    print(max_k)

solve()