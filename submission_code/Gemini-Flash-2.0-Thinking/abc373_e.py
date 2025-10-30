def solve():
    n, m, k = map(int, input().split())
    a = list(map(int, input().split()))

    remaining_votes = k - sum(a)
    results = []

    for i in range(n):
        if sum(1 for j in range(n) if i != j and a[j] > a[i]) < m:
            results.append(0)
            continue

        ans_i = -1
        low = 0
        high = remaining_votes

        while low <= high:
            mid_x = (low + high) // 2
            target_votes_i = a[i] + mid_x
            remaining_dist = remaining_votes - mid_x
            num_greater = 0

            temp_a = list(a)
            temp_a[i] = target_votes_i

            can_exceed = []
            for j in range(n):
                if i != j:
                    if temp_a[j] > target_votes_i:
                        num_greater += 1
                    else:
                        can_exceed.append(j)

            sorted_can_exceed = sorted(can_exceed, key=lambda x: temp_a[x])

            temp_remaining_dist = remaining_dist

            for j_idx in sorted_can_exceed:
                needed = target_votes_i + 1 - temp_a[j_idx]
                if temp_remaining_dist >= needed:
                    num_greater += 1
                    temp_remaining_dist -= needed

            if num_greater < m:
                ans_i = mid_x
                high = mid_x - 1
            else:
                low = mid_x + 1

        results.append(ans_i)

    print(*results)

solve()