def solve():
    n, k = map(int, input().split())
    p = list(map(int, input().split()))

    def apply_op(perm):
        n = len(perm)
        new_perm = [0] * n
        for i in range(n):
            new_perm[i] = perm[perm[i] - 1]
        return new_perm

    def power_op(perm, k):
        n = len(perm)
        if k == 0:
            return list(range(1, n + 1))
        if k == 1:
            return perm

        temp_perm = list(perm)
        result_perm = list(range(1, n + 1))

        while k > 0:
            if k % 2 == 1:
                new_result = [0] * n
                for i in range(n):
                    new_result[i] = result_perm[temp_perm[i] - 1]
                result_perm = new_result

            new_temp = [0] * n
            for i in range(n):
                new_temp[i] = temp_perm[temp_perm[i] - 1]
            temp_perm = new_temp
            k //= 2
        return result_perm

    final_p = power_op(p, k)
    print(*final_p)

solve()