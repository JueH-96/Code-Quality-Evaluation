def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    # Parse inputs:
    N, M, K = map(int, input_data[:3])
    index = 3

    # We will store each test as (mask, result),
    # where mask is a bitmask representing which keys are used,
    # and result is True (opened) or False (did not open).
    tests = []
    for _ in range(M):
        C_i = int(input_data[index]); index += 1
        # Compute bitmask
        mask = 0
        for _c in range(C_i):
            key_id = int(input_data[index]); index += 1
            mask |= (1 << (key_id - 1))
        R_i = input_data[index]; index += 1
        tests.append((mask, (R_i == 'o')))  # True if 'o', else False

    # We'll brute force all subsets of {1..N}.
    # For each subset s, check all tests.
    # If for test i, popcount(s & mask) >= K then the door would open => test result must be 'o'.
    # Else test result must be 'x'.
    # Count how many such s pass all M tests.
    
    # Precompute popcount for all relevant intersections might be large,
    # but Python's built-in bit_count() is fast enough for N <= 15.

    valid_count = 0
    ALL = 1 << N  # total number of subsets
    for s in range(ALL):
        is_valid = True
        for mask, opened in tests:
            # Count real keys in intersection:
            real_count = (s & mask).bit_count()
            door_open_this_subset = (real_count >= K)
            if door_open_this_subset != opened:
                is_valid = False
                break
        if is_valid:
            valid_count += 1

    print(valid_count)

def main():
    solve()

if __name__ == "__main__":
    main()