def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = input[1].strip()
    length = 3 ** N
    assert len(A) == length

    # Initialize level N
    prev_current_val = [int(c) for c in A]
    prev_cost0 = [0 if v == 0 else 1 for v in prev_current_val]
    prev_cost1 = [0 if v == 1 else 1 for v in prev_current_val]

    all_combinations = [(t0, t1, t2) for t0 in (0, 1) for t1 in (0, 1) for t2 in (0, 1)]

    for k in range(N-1, -1, -1):
        current_size = 3 ** k
        current_val = [0] * current_size
        current_cost0 = [float('inf')] * current_size
        current_cost1 = [float('inf')] * current_size

        for i in range(current_size):
            a = 3 * i
            b = a + 1
            c = a + 2

            # Get children's current values
            c0 = prev_current_val[a]
            c1 = prev_current_val[b]
            c2 = prev_current_val[c]

            # Compute current_val[i]
            sum_ones = c0 + c1 + c2
            current_val[i] = 1 if sum_ones >= 2 else 0

            # Compute cost0 and cost1 for this node
            min0 = float('inf')
            min1 = float('inf')

            for (t0, t1, t2) in all_combinations:
                sum_t = t0 + t1 + t2
                cost = 0
                cost += (prev_cost0[a] if t0 == 0 else prev_cost1[a])
                cost += (prev_cost0[b] if t1 == 0 else prev_cost1[b])
                cost += (prev_cost0[c] if t2 == 0 else prev_cost1[c])

                if sum_t < 2 and cost < min0:
                    min0 = cost
                if sum_t >= 2 and cost < min1:
                    min1 = cost

            current_cost0[i] = min0
            current_cost1[i] = min1

        # Update previous arrays for next iteration
        prev_current_val = current_val
        prev_cost0 = current_cost0
        prev_cost1 = current_cost1

    # The root is at level 0, index 0
    current_root_val = prev_current_val[0]
    desired_val = 1 - current_root_val
    if desired_val == 0:
        print(prev_cost0[0])
    else:
        print(prev_cost1[0])

if __name__ == "__main__":
    main()