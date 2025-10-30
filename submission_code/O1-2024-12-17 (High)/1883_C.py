def main():
    import sys

    # Precompute the exponent of 2 (e2) for numbers up to 40 (more than enough for our increments).
    MAX_VAL = 40
    e2_table = [0] * (MAX_VAL + 1)
    for x in range(MAX_VAL + 1):
        cnt = 0
        temp = x
        while temp > 0 and temp % 2 == 0:
            cnt += 1
            temp //= 2
        e2_table[x] = cnt

    # Precompute cost_gain_1 and cost_gain_2 for all a in [1..10].
    # cost_gain_1[a] = minimum increments needed so that e2(a + d) - e2(a) >= 1
    # cost_gain_2[a] = minimum increments needed so that e2(a + d) - e2(a) >= 2
    INF = 10**9
    cost_gain_1 = [INF] * 11
    cost_gain_2 = [INF] * 11
    for a in range(1, 11):
        base_e2 = e2_table[a]
        best1 = INF
        best2 = INF
        # We'll allow up to 20 increments (which covers 1..10 plus up to 30).
        for d in range(21):
            val = a + d
            if val > MAX_VAL:
                break
            net_gain = e2_table[val] - base_e2
            if net_gain >= 1 and d < best1:
                best1 = d
            if net_gain >= 2 and d < best2:
                best2 = d
        cost_gain_1[a] = best1
        cost_gain_2[a] = best2

    input_data = sys.stdin.read().strip().split()
    t = int(input_data[0])
    idx = 1

    answers = []
    for _ in range(t):
        n = int(input_data[idx]); idx += 1
        k = int(input_data[idx]); idx += 1
        arr = list(map(int, input_data[idx:idx+n]))
        idx += n

        if k == 2:
            # We need at least one even number in the product.
            # If there's already an even number => 0 operations.
            # Otherwise => 1 operation (increment any odd number once to make it even).
            has_even = any((x % 2 == 0) for x in arr)
            answers.append('0' if has_even else '1')

        elif k == 3:
            # We need at least one multiple of 3 in the product.
            # If there's already a multiple of 3 => 0 operations.
            # Otherwise => min(3 - x%3).
            has_mult_3 = any((x % 3 == 0) for x in arr)
            if has_mult_3:
                answers.append('0')
            else:
                # Compute minimal increments to reach a multiple of 3
                best = min((3 - (x % 3)) for x in arr)
                answers.append(str(best))

        elif k == 5:
            # We need at least one multiple of 5.
            has_mult_5 = any((x % 5 == 0) for x in arr)
            if has_mult_5:
                answers.append('0')
            else:
                # Minimal increments to reach a multiple of 5
                best = min((5 - (x % 5)) for x in arr)
                answers.append(str(best))

        else:
            # k == 4 => we need the product to have at least 2 factors of 2 in total.
            # Sum up the exponents of 2 in all elements, if >= 2 => 0 ops.
            # If sum is 1 => we need 1 more => pick min cost_gain_1.
            # If sum is 0 => either pick one element to gain 2 exponents or pick two elements to gain 1 exponent each.
            s2 = sum(e2_table[x] for x in arr)
            if s2 >= 2:
                answers.append('0')
            elif s2 == 1:
                # Need exactly +1 exponent
                best = min(cost_gain_1[x] for x in arr)
                answers.append(str(best))
            else:
                # s2 == 0, need +2 exponents
                # Option A: get +2 from a single element
                best_single = min(cost_gain_2[x] for x in arr)
                # Option B: get +1 from two distinct elements => pick two smallest cost_gain_1
                # We'll gather cost_gain_1 for all elements and take the two smallest.
                c1_vals = [cost_gain_1[x] for x in arr]
                c1_vals.sort()
                if len(c1_vals) >= 2:
                    best_double = c1_vals[0] + c1_vals[1]
                else:
                    best_double = INF

                ans = min(best_single, best_double)
                answers.append(str(ans))

    print('
'.join(answers))


# Do not forget to call main() at the end!
if __name__ == "__main__":
    main()