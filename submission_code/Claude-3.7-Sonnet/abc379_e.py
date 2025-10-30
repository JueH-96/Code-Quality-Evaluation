def solve():
    N = int(input())
    S = input()
    
    total_sum = 0
    
    # Precompute the repunits (10^i - 1) / 9 for i from 1 to N
    repunit = 1  # Start with (10^1 - 1) / 9 = 1
    repunits = [repunit]
    
    for i in range(2, N + 1):
        repunit = repunit * 10 + 1
        repunits.append(repunit)
    
    for k in range(N):
        # Convert the character to an integer
        digit = int(S[k])
        
        # Calculate the contribution of this digit to the total sum
        contribution = digit * (k + 1) * repunits[N-k-1]
        
        total_sum += contribution
    
    print(total_sum)

solve()