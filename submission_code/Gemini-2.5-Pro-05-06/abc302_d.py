import bisect
import sys

def main():
    N, M, D = map(int, sys.stdin.readline().split())
    A_list = list(map(int, sys.stdin.readline().split()))
    B_list = list(map(int, sys.stdin.readline().split()))

    A_list.sort()
    B_list.sort()

    max_sum = -1

    # Decide which list to iterate and which to search in.
    # Iterate through the (potentially) shorter list and binary search in the (potentially) longer one.
    # This is a micro-optimization; iterating A and searching B would also pass.
    
    iter_list, search_list = A_list, B_list
    if N > M: # If A_list is longer than B_list, iterate B_list and search A_list
        iter_list, search_list = B_list, A_list

    for val1 in iter_list:
        # For current val1, we want to find val2 in search_list such that:
        # val1 - D <= val2 <= val1 + D
        # To maximize (val1 + val2), we need to maximize val2.
        # So, we look for the largest val2 in search_list satisfying the conditions.
        
        # Upper bound for val2 is val1 + D
        target_upper_bound_for_val2 = val1 + D
        
        # bisect_right finds an insertion point `idx` which comes after (to the right of) any existing entries
        # of `target_upper_bound_for_val2` and maintains sorted order.
        # Thus, all elements search_list[j] for j < idx satisfy search_list[j] <= target_upper_bound_for_val2.
        # If idx > 0, search_list[idx-1] is the largest element in search_list that is <= target_upper_bound_for_val2.
        idx = bisect.bisect_right(search_list, target_upper_bound_for_val2)
        
        if idx == 0:
            # This means all elements in search_list are > target_upper_bound_for_val2.
            # (Or search_list is empty, but constraints say N, M >= 1).
            # No suitable val2 can be found for this val1.
            continue
            
        val2_candidate = search_list[idx-1] # This is the largest element in search_list s.t. val2_candidate <= val1 + D
        
        # Now, we must check if this val2_candidate also satisfies the lower bound:
        # val2_candidate >= val1 - D
        # This check, combined with val2_candidate <= val1 + D, ensures |val1 - val2_candidate| <= D.
        
        if val2_candidate >= val1 - D:
            # val2_candidate is a valid partner for val1.
            current_sum = val1 + val2_candidate
            if current_sum > max_sum:
                max_sum = current_sum
    
    sys.stdout.write(str(max_sum) + "
")

if __name__ == '__main__':
    main()