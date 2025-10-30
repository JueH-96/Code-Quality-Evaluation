def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    S = int(input[1])
    A = list(map(int, input[2:2+N]))
    
    # Case a: check for any subarray sum equals S
    prefix_sum = 0
    seen = {0}
    found = False
    for num in A:
        prefix_sum += num
        if (prefix_sum - S) in seen:
            found = True
            break
        seen.add(prefix_sum)
    if found:
        print("Yes")
        return
    
    sum_A = sum(A)
    S_mod = S % sum_A
    
    # Compute suffix_rem
    suffix_rem = {}
    current_sum = 0
    for i in range(N-1, -1, -1):
        current_sum += A[i]
        rem = current_sum % sum_A
        if rem not in suffix_rem:
            suffix_rem[rem] = (current_sum, current_sum)
        else:
            old_min, old_max = suffix_rem[rem]
            new_min = min(old_min, current_sum)
            new_max = max(old_max, current_sum)
            suffix_rem[rem] = (new_min, new_max)
    
    # Compute prefix_rem
    prefix_rem = {}
    current_sum = 0
    for num in A:
        current_sum += num
        rem = current_sum % sum_A
        if rem not in prefix_rem:
            prefix_rem[rem] = (current_sum, current_sum)
        else:
            old_min, old_max = prefix_rem[rem]
            new_min = min(old_min, current_sum)
            new_max = max(old_max, current_sum)
            prefix_rem[rem] = (new_min, new_max)
    
    # Check for case b
    for s_rem in suffix_rem:
        required_p_rem = (S_mod - s_rem) % sum_A
        if required_p_rem in prefix_rem:
            min_s = suffix_rem[s_rem][0]
            min_p = prefix_rem[required_p_rem][0]
            if min_s + min_p <= S:
                print("Yes")
                return
    
    print("No")

if __name__ == "__main__":
    main()