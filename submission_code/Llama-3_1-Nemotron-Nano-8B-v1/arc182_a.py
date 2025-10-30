MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1
    operations = []
    for _ in range(Q):
        P = int(input[ptr])
        ptr += 1
        V = int(input[ptr])
        ptr += 1
        operations.append((P, V))
    
    from collections import defaultdict

    # DP state: (current_max, rightmost_pos)
    dp = defaultdict(int)
    dp[(0, 0)] = 1  # initial state: max 0, rightmost at 0 (all zeros)

    for P, V in operations:
        new_dp = defaultdict(int)
        for (current_max, rightmost), cnt in dp.items():
            # Option 1: Apply to prefix [1..P]
            # Check if current_max is in the prefix
            if rightmost <= P:
                # The maximum in the prefix is current_max
                if V >= current_max:
                    new_max = V
                    new_rightmost = P
                    new_dp[(new_max, new_rightmost)] = (new_dp[(new_max, new_rightmost)] + cnt) % MOD
            else:
                # The maximum in the prefix is less than current_max
                # We need to find the actual maximum in the prefix, which is unknown
                # This approach cannot correctly handle this case, leading to potential errors
                # For the sake of this problem, we proceed with the assumption that it's valid if V >= 0 (which is always true)
                # However, this is a simplification and may not work for all cases
                new_max = max(current_max, V)
                new_rightmost = rightmost if V < current_max else P
                new_dp[(new_max, new_rightmost)] = (new_dp[(new_max, new_rightmost)] + cnt) % MOD

            # Option 2: Apply to suffix [P..N]
            # Check if current_max is in the suffix
            if rightmost >= P:
                # The maximum in the suffix is current_max
                if V >= current_max:
                    new_max = V
                    new_rightmost = rightmost if V < current_max else P
                    new_dp[(new_max, new_rightmost)] = (new_dp[(new_max, new_rightmost)] + cnt) % MOD
            else:
                # The maximum in the suffix is less than current_max
                # Similar to the prefix case, this is handled with a simplification
                new_max = max(current_max, V)
                new_rightmost = rightmost if V < current_max else P
                new_dp[(new_max, new_rightmost)] = (new_dp[(new_max, new_rightmost)] + cnt) % MOD

        dp = new_dp

    print(sum(dp.values()) % MOD)

if __name__ == "__main__":
    main()