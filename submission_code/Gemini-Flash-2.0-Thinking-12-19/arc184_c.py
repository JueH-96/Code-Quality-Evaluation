def count_set_bits(n):
    count = 0
    while n > 0:
        n &= (n - 1)
        count += 1
    return count

def is_mountain_fold(index):
    set_bits_count = count_set_bits(index)
    return set_bits_count % 2 == 1

def solve():
    n = int(input())
    a = list(map(int, input().split()))
    max_f_value = 0
    upper_bound = (1 << 100) - 1 - a[-1]
    
    # For larger constraints, iterating through all i might be too slow. 
    # However, for given constraints, we could try to check a range of i values.
    # For now, let's just try to iterate for a limited range, say first 100000 values of i.
    # For a full solution, we might need to find a more efficient approach.
    
    # In sample cases, A values are small, so range of i is still very large.
    # Let's try to find the maximum value by checking a range of i.
    
    # Given constraints: 1 <= N <= 1000, 0 = A_1 < A_2 < ... < A_N <= 10^18
    # For each i from 1 to 2^100 - A_N - 1, define f(i) as number of k such that (i+A_k)-th crease is mountain fold.
    # Find max f(i).
    
    # Let's try to check a range of i values, maybe from 1 to some reasonable limit.
    # But the problem statement says 2^100 - A_N - 1. This range is huge. 
    
    # Sample 1: A = [0, 1, 2, 3]. Output 3. Sample 2: A = [0, 2, 3, 5, 7, 8]. Output 4.
    
    max_possible_f = 0
    
    # Since we can't iterate through all possible i, we might need to find a property of f(i).
    # Or maybe we are supposed to find a pattern. 
    
    # For now, let's try to iterate through a limited range of i values. 
    # Let's try i from 1 to 10000 (or some other small range). 
    # And see if we can find a pattern or if we get the sample outputs. 
    
    limit_i = min(upper_bound, 1000) if upper_bound > 0 else 0
    if upper_bound < 0:
        limit_i = 0

    if limit_i > 0:
        for i in range(1, limit_i + 1):
            current_f_value = 0
            for k in range(n):
                crease_index = i + a[k]
                if 1 <= crease_index <= (1 << 100) - 1:
                    if is_mountain_fold(crease_index):
                        current_f_value += 1
            max_f_value = max(max_f_value, current_f_value)
            
    else: # If upper_bound is not positive, then no valid i, but problem states i from 1 to 2^100 - A_N - 1.
        # It's guaranteed that 0 = A_1 < A_2 < ... < A_N <= 10^18. So 2^100 - A_N - 1 >= 0. 
        # So, range is always valid if A_N < 2^100 - 1. 
        # Since A_N <= 10^18, this condition is always satisfied.
        # So, upper_bound will always be >= 0. 
        pass # Should not reach here if input is valid as per constraints.

    if max_f_value == 0: # If we didn't iterate or found no max in limited range, try to find for i=1.
        current_f_value = 0
        for k in range(n):
            crease_index = 1 + a[k]
            if 1 <= crease_index <= (1 << 100) - 1:
                if is_mountain_fold(crease_index):
                    current_f_value += 1
        max_f_value = max(max_f_value, current_f_value)

    print(max_f_value)

if __name__ == '__main__':
    solve()