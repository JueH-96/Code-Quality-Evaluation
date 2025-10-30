def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    K = int(input[idx])
    idx += 1

    group_x = []
    group_y = []

    for _ in range(N):
        a = int(input[idx])
        idx += 1
        b = int(input[idx])
        idx += 1
        if a == 1:
            group_y.append(b)
        else:
            group_x.append((a, b))

    # Sort group_y in descending order of B
    group_y.sort(reverse=True)

    max_total = 0

    # Iterate over possible m (number of functions from group_x)
    for m in range(0, K + 1):
        required_y = K - m
        if required_y < 0:
            continue
        # Check if we have enough functions in group_x and group_y
        if m > len(group_x) or required_y > len(group_y):
            continue

        # Compute sum_by
        sum_by = sum(group_y[:required_y]) if required_y > 0 else 0

        # Handle group_x selection
        if m == 0:
            product_a = 1
            sum_b = 0
        else:
            # Sort group_x based on the key for current sum_by
            sorted_x = sorted(group_x, key=lambda x: -(x[0] * (sum_by + 1) + x[1]))
            selected = sorted_x[:m]
            # Compute product_a and sum_b by processing in reverse order
            product_a = 1
            sum_b = 0
            for a, b in reversed(selected):
                sum_b = sum_b * a + b
                product_a *= a

        total = product_a * (sum_by + 1) + sum_b
        if total > max_total:
            max_total = total

    print(max_total)

if __name__ == "__main__":
    main()