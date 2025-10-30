def min_operations(nums: List[int]) -> int:
    max_n = max(nums) if nums else 0
    if max_n == 0:
        return 0  # Edge case, but per problem statement, nums contain positive integers
    
    # Precompute smallest prime factors up to max_n
    spf = [0] * (max_n + 1)
    for i in range(2, max_n + 1):
        if spf[i] == 0:
            spf[i] = i
            if i * i <= max_n:
                for j in range(i * i, max_n + 1, i):
                    if spf[j] == 0:
                        spf[j] = i
    
    possible_values = []
    steps = []
    
    for x in nums:
        if x == 1:
            possible = [1]
            steps_i = [0]
        else:
            spf_x = spf[x]
            g = x // spf_x
            current = x
            pv = []
            while True:
                pv.append(current)
                if current == 1:
                    break
                current = current // g
                if current >= x or current <= 0:
                    break
            possible = pv
            steps_i = [0]
            for i in range(len(pv) - 1):
                steps_i.append(steps_i[-1] + 1)
        possible_values.append(possible)
        steps.append(steps_i)
    
    current_max = float('inf')
    total_ops = 0
    
    for i in range(len(nums) - 1, -1, -1):
        pv = possible_values[i]
        steps_i = steps[i]
        
        if i == len(nums) - 1:
            max_val = pv[0]
            total_ops += steps_i[0]
            current_max = max_val
        else:
            found = False
            for j in range(len(pv)):
                if pv[j] <= current_max:
                    total_ops += steps_i[j]
                    current_max = pv[j]
                    found = True
                    break
            if not found:
                return -1
    
    return total_ops