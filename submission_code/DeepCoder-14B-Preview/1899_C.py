def max_alternating_parity_subarray_sum():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    t = int(input[idx])
    idx += 1
    for _ in range(t):
        n = int(input[idx])
        idx += 1
        a = list(map(int, input[idx:idx+n]))
        idx += n
        
        max_sum = -float('inf')
        current_even = None
        current_odd = None
        
        for num in a:
            if num % 2 == 0:
                # Current number is even
                if current_odd is not None:
                    temp_even = max(num, current_odd + num)
                else:
                    temp_even = num
                current_even = temp_even
                current_odd = None  # Subarrays ending here can't end with odd
            else:
                # Current number is odd
                if current_even is not None:
                    temp_odd = max(num, current_even + num)
                else:
                    temp_odd = num
                current_odd = temp_odd
                current_even = None  # Subarrays ending here can't end with even
            
            # Determine the current maximum
            if current_even is not None:
                current_max = current_even
            else:
                current_max = current_odd
            
            if current_max > max_sum:
                max_sum = current_max
        
        print(max_sum)

max_alternating_parity_subarray_sum()