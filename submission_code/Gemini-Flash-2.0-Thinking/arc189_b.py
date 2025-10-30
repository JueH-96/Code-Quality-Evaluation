def solve():
    n = int(input())
    x = list(map(int, input().split()))

    def apply_operation(arr, i):
        m = (arr[i-1] + arr[i+2]) / 2
        new_xi = 2 * m - arr[i]
        new_xi_plus_1 = 2 * m - arr[i+1]
        new_arr = list(arr)
        new_arr[i] = new_xi
        new_arr[i+1] = new_xi_plus_1
        return sorted(new_arr)

    def calculate_sum(arr):
        return sum(arr)

    min_sum = calculate_sum(x)
    q = [list(x)]
    seen_sums = {min_sum}

    while q:
        current_x = q.pop(0)
        current_sum = calculate_sum(current_x)

        for i in range(1, n - 2):
            next_x = apply_operation(current_x, i)
            next_sum = calculate_sum(next_x)

            if next_sum < min_sum:
                min_sum = next_sum

            if next_sum not in seen_sums:
                seen_sums.add(next_sum)
                q.append(next_x)

    print(min_sum)

solve()