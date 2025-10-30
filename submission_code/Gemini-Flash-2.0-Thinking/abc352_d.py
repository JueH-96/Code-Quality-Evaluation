def solve():
    n, k = map(int, input().split())
    p = list(map(int, input().split()))

    min_diff = float('inf')

    pos = [0] * (n + 1)
    for i in range(n):
        pos[p[i]] = i

    for start_val in range(1, n - k + 2):
        consecutive_elements = list(range(start_val, start_val + k))
        indices = []
        possible = True
        for val in consecutive_elements:
            if 1 <= val <= n:
                indices.append(pos[val])
            else:
                possible = False
                break

        if possible:
            indices_with_original_order = sorted(enumerate(indices), key=lambda x: x[1])
            sorted_indices = [item[1] for item in indices_with_original_order]

            is_increasing = all(sorted_indices[i] < sorted_indices[i+1] for i in range(k-1))

            if is_increasing:
                original_indices = sorted([item[0] + 1 for item in indices_with_original_order])
                diff = original_indices[-1] - original_indices[0]
                min_diff = min(min_diff, diff)

    print(min_diff)

solve()