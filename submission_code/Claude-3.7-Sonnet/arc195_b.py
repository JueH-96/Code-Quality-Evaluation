def solve():
    N = int(input())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    
    # Extract fixed elements (not -1)
    fixed_A = [a for a in A if a != -1]
    fixed_B = [b for b in B if b != -1]
    
    # If there are no fixed elements, we can set all to 0
    if not fixed_A and not fixed_B:
        return "Yes"
    
    # Gather all potential target sums
    potential_sums = set()
    
    # From existing fixed pairs
    for i in range(N):
        if A[i] != -1 and B[i] != -1:
            potential_sums.add(A[i] + B[i])
    
    # From possible pairings of fixed elements
    for a in fixed_A:
        for b in fixed_B:
            potential_sums.add(a + b)
    
    # If no potential sums (shouldn't happen with our logic), default to 0
    if not potential_sums:
        potential_sums = {0}
    
    # Try each potential sum
    for target_sum in potential_sums:
        can_achieve = True
        
        # Check if all fixed B elements can be paired with valid A elements
        required_A_freq = {}
        for b in fixed_B:
            required_val = target_sum - b
            if required_val < 0:  # Would need negative A, not allowed
                can_achieve = False
                break
            required_A_freq[required_val] = required_A_freq.get(required_val, 0) + 1
        
        if not can_achieve:
            continue
        
        # Check if all fixed A elements can be paired with valid B elements
        required_B_freq = {}
        for a in fixed_A:
            required_val = target_sum - a
            if required_val < 0:  # Would need negative B, not allowed
                can_achieve = False
                break
            required_B_freq[required_val] = required_B_freq.get(required_val, 0) + 1
        
        if not can_achieve:
            continue
        
        # Calculate how many of each value we already have in fixed A
        available_A_freq = {}
        for a in fixed_A:
            available_A_freq[a] = available_A_freq.get(a, 0) + 1
        
        # Count how many -1's in A we need to replace
        unfilled_A_count = 0
        for val, count in required_A_freq.items():
            unfilled_A_count += max(0, count - available_A_freq.get(val, 0))
        
        if unfilled_A_count > A.count(-1):  # Not enough -1's in A
            continue
        
        # Calculate how many of each value we already have in fixed B
        available_B_freq = {}
        for b in fixed_B:
            available_B_freq[b] = available_B_freq.get(b, 0) + 1
        
        # Count how many -1's in B we need to replace
        unfilled_B_count = 0
        for val, count in required_B_freq.items():
            unfilled_B_count += max(0, count - available_B_freq.get(val, 0))
        
        if unfilled_B_count > B.count(-1):  # Not enough -1's in B
            continue
        
        # If we reach here, all conditions are satisfied
        return "Yes"
    
    # No valid target sum found
    return "No"

print(solve())