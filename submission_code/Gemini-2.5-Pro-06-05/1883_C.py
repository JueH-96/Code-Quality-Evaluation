import sys

def solve():
    """
    Solves a single test case by analyzing the value of k.
    """
    try:
        n, k = map(int, sys.stdin.readline().split())
        a = list(map(int, sys.stdin.readline().split()))
    except (IOError, ValueError):
        # Handles empty lines or malformed input at the end of file
        return

    if k == 4:
        # We need the product to be divisible by 4, which means we need
        # a total of at least two factors of 2 from the numbers.
        # A multiple of 4 contributes two factors. An even number contributes one.
        
        evens_count = 0
        has_mult_of_4 = False
        for x in a:
            if x % 4 == 0:
                has_mult_of_4 = True
                break
            elif x % 2 == 0:
                evens_count += 1
        
        if has_mult_of_4 or evens_count >= 2:
            # The product is already divisible by 4.
            print(0)
            return

        # We need more factors of 2.
        # Case 1: We have one even number (so one factor of 2).
        if evens_count == 1:
            # We need one more factor. The cheapest way is to increment an odd number
            # by 1 to make it even. Cost is 1.
            print(1)
        # Case 2: All numbers are odd (zero factors of 2).
        else:
            # We need two factors.
            # Option A: Make one number a multiple of 4.
            cost_to_make_one_mult_4 = float('inf')
            for x in a:
                # Cost to make x a multiple of 4. For an odd x, this is 1 if x%4==3, else 3.
                cost_to_make_one_mult_4 = min(cost_to_make_one_mult_4, (4 - x % 4) % 4)
            
            # Option B: Make two different numbers even. Cost is 1+1=2.
            cost_to_make_two_evens = 2
            
            print(min(cost_to_make_one_mult_4, cost_to_make_two_evens))

    else: # k is 2, 3, or 5 (prime)
        # We need at least one number to be a multiple of k.
        min_ops = float('inf')
        for x in a:
            # Cost to make x the next multiple of k.
            # (k - x % k) % k handles the case where x is already a multiple.
            ops_for_x = (k - x % k) % k
            min_ops = min(min_ops, ops_for_x)
        print(min_ops)


def main():
    """
    Main function to handle multiple test cases.
    """
    try:
        num_test_cases_str = sys.stdin.readline()
        if not num_test_cases_str: return
        num_test_cases = int(num_test_cases_str)
        for _ in range(num_test_cases):
            solve()
    except (IOError, ValueError):
        return

if __name__ == "__main__":
    main()