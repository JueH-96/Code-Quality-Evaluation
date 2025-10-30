import collections
import sys

def solve_single_test_case():
    n = int(sys.stdin.readline())
    a = list(map(int, sys.stdin.readline().split()))

    counts = collections.Counter(a)
    
    ans = 0
    
    # Case 1: Count pairs (i, j) with i < j such that a_i = a_j.
    # For each distinct value 'val' that appears 'count_of_val' times,
    # it contributes count_of_val * (count_of_val - 1) / 2 pairs.
    for val_key in counts:
        count_of_val = counts[val_key]
        if count_of_val >= 2:
            ans += count_of_val * (count_of_val - 1) // 2
            
    # Case 2: Count pairs (i, j) with i < j such that {a_i, a_j} = {1, 2}.
    # These pairs are not covered by Case 1 (where a_i = a_j).
    # For example, if a_i=1 and a_j=1, it's Case 1. If a_i=2 and a_j=2, it's Case 1.
    # If a_i=1 and a_j=2 (or vice-versa), it's Case 2.
    
    # Get count of 1s and 2s. collections.Counter returns 0 for keys not present.
    count_1 = counts[1]
    count_2 = counts[2]
    
    # Each occurrence of 1 can be paired with each occurrence of 2.
    # This forms count_1 * count_2 such pairs.
    # If either count_1 or count_2 is 0, this product is 0, so ans is not affected.
    ans += count_1 * count_2
    
    sys.stdout.write(str(ans) + "
")

def main():
    num_test_cases = int(sys.stdin.readline())
    for _ in range(num_test_cases):
        solve_single_test_case()

if __name__ == '__main__':
    main()