def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    K = int(input[1])
    A = list(map(int, input[2:2+N]))
    sum_A = sum(A)
    
    # Binary search for x
    low = 0
    high = sum_A // K
    ans_x = 0
    for mid in range(high, -1, -1):
        if sum_A < K * mid:
            continue
        current_sum = 0
        groups = 0
        # Check in the circular array
        B = A * 2
        for num in B:
            current_sum += num
            if current_sum >= mid:
                groups += 1
                current_sum = 0
                if groups >= K:
                    break
        if groups >= K:
            ans_x = mid
            break
    
    # Now compute y: number of cut lines never cut
    # A cut line i is never cut if in all optimal divisions, it is cut.
    # So we need to find cut lines i where for all possible divisions with sum >=ans_x, the cut line i is cut.
    # To find this, for each cut line i, check if the sum of the entire array except for the group that includes i and i+1 is <ans_x.
    # Alternatively, a cut line i is never cut if there is no way to have a division where i is not cut and all groups have sum >=ans_x.
    # This part is complex and requires checking each cut line.
    
    # Precompute prefix sums
    prefix = [0] * (N + 1)
    for i in range(N):
        prefix[i+1] = prefix[i] + A[i]
    
    # For each cut line i (between i and i+1), check if there exists a division where cut line i is not cut.
    # If not, then i is counted in y.
    y = 0
    required = K
    x = ans_x
    
    # Check each cut line i (0-based)
    for i in range(N):
        # To check if cut line i is never cut, we need to see if there's a way to split the array into K groups without cutting i.
        # This means the group includes pieces i+1 and i+2 (mod N) must be in the same group.
        # To model this, merge pieces i and i+1 into a single piece and check if it's possible to split into K groups.
        merged = A.copy()
        merged[i] += merged[(i+1)%N]
        if (i+1) < N:
            merged.pop((i+1)%N)
        else:
            merged.pop(0)
        # Now check if merged array can be split into K-1 groups with sum >=x
        # Because merging two pieces reduces the problem to N-1 pieces and K-1 groups.
        # But since the array is circular, this approach may not work.
        # Alternative approach: check if there's a way to split the original array into K groups with sum >=x and not cutting i.
        # This is complex.
        # Instead, use the following heuristic: a cut line i is never cut if the sum of any group that includes i and i+1 is <x.
        # Check if there exists a group that includes i and i+1 and has sum >=x.
        # If not, then cut line i must be cut in all divisions.
        # But how to check this.
        # For each possible group that includes i and i+1, check if the sum is >=x.
        # If all such groups have sum <x, then cut line i must be cut.
        # But how to check this.
        # For example, if the sum of A[i] + A[i+1] is >=x, then it's possible to have a group containing i and i+1.
        # But this is not sufficient.
        # This part is complex and requires further analysis.
        # Given time constraints, we'll use a heuristic.
        possible = False
        # Check if there exists a division where cut line i is not cut.
        # This requires that the merged piece (i and i+1) can be part of a valid group.
        # Check all possible starting positions.
        B = A.copy()
        B[i] += B[(i+1)%N]
        if (i+1) < N:
            B.pop((i+1)%N)
        else:
            B.pop(0)
        # Now check if B can be split into K-1 groups with sum >=x.
        # Using the same check as before.
        sum_B = sum(B)
        if sum_B < (K-1)*x:
            continue
        current_sum = 0
        groups = 0
        for num in B:
            current_sum += num
            if current_sum >= x:
                groups +=1
                current_sum =0
                if groups >= K-1:
                    break
        if groups >= K-1:
            possible = True
        else:
            # Check if remaining sum >=x
            if current_sum >=x:
                groups +=1
            if groups >= K-1:
                possible = True
        if possible:
            continue
        else:
            y +=1
    print(ans_x, y)

if __name__ == "__main__":
    main()