def main():
    import sys
    
    input_data = sys.stdin.read().strip().split()
    t = int(input_data[0])
    ptr = 1

    # Precompute the number of factors of 2 for all numbers up to 30 or so (enough to handle increments)
    # (We only really need up to 22 or a bit more, but 30 is a small overhead.)
    MAX_VAL = 30
    count2 = [0]*(MAX_VAL+1)
    for x in range(1, MAX_VAL+1):
        tmp = x
        c = 0
        while tmp % 2 == 0:
            tmp >>= 1
            c += 1
        count2[x] = c

    out = []
    for _ in range(t):
        n = int(input_data[ptr]); ptr += 1
        k = int(input_data[ptr]); ptr += 1
        arr = list(map(int, input_data[ptr:ptr+n]))
        ptr += n

        # Case for k = 2, 3, 5 are straightforward
        if k == 2:
            # Need the product to be even => at least one element even
            # If there's already an even => 0, else => 1
            ans = 0
            if all(a % 2 != 0 for a in arr):
                ans = 1
            out.append(str(ans))
            continue

        if k == 3:
            # Need at least one multiple of 3
            # If any a_i % 3 == 0 => 0, else => min( (3 - a_i%3) %3 )
            if any(a % 3 == 0 for a in arr):
                out.append('0')
                continue
            else:
                ans = min((3 - (a % 3)) % 3 for a in arr)
                out.append(str(ans))
                continue

        if k == 5:
            # Need at least one multiple of 5
            # If any a_i % 5 == 0 => 0, else => min( (5 - a_i%5) %5 )
            if any(a % 5 == 0 for a in arr):
                out.append('0')
                continue
            else:
                ans = min((5 - (a % 5)) % 5 for a in arr)
                out.append(str(ans))
                continue

        # The remaining interesting case is k = 4
        # We need at least 2 factors of 2 in the product
        # Count how many total factors of 2 are already present
        total_twos = 0
        for a in arr:
            total_twos += count2[a]

        if total_twos >= 2:
            # Already enough twos
            out.append('0')
            continue

        # Otherwise, we need to add more factors of 2 by incrementing
        needed = 2 - total_twos  # will be 1 or 2

        # Precompute, for each element, the minimal cost to gain at least +1 or +2 new factors.
        INF = 10**9
        plus_one = [INF]*n
        plus_two = [INF]*n

        for i, val in enumerate(arr):
            orig_2 = count2[val]
            # Try costs up to some small limit (say 13 to be safe)
            for cost in range(14):
                new_val = val + cost
                if new_val <= MAX_VAL:
                    new_2 = count2[new_val]
                else:
                    # If we ever go beyond our precomputed array, just compute quickly
                    tmp = new_val
                    c2 = 0
                    while tmp % 2 == 0:
                        tmp >>= 1
                        c2 += 1
                    new_2 = c2

                inc = new_2 - orig_2
                if inc >= 1 and cost < plus_one[i]:
                    plus_one[i] = cost
                if inc >= 2 and cost < plus_two[i]:
                    plus_two[i] = cost

        if needed == 1:
            # We just need +1 factor in total
            ans = min(plus_one)
            out.append(str(ans))
        else:
            # needed == 2
            # Either one element can give +2 in one go (min(plus_two))
            # or we combine the two smallest distinct plus_one costs
            best_two = min(plus_two)
            # Sort the plus_one array (cost, index)
            arr_plus_one = sorted((c, i) for i, c in enumerate(plus_one))
            # The cheapest way to pick two distinct elements for +1 each
            # is just the sum of the first two costs in this sorted list
            # (since there's exactly one entry per element, the first two are distinct)
            if n >= 2:
                sum_two_ones = arr_plus_one[0][0] + arr_plus_one[1][0]
            else:
                # theoretically n>=2 is guaranteed by problem statement
                sum_two_ones = INF

            ans = best_two if best_two < sum_two_ones else sum_two_ones
            out.append(str(ans))

    print("
".join(out))

# Do not forget to call main()
if __name__ == "__main__":
    main()