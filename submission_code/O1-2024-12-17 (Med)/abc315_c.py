def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    
    N = int(input_data[0])
    FT = input_data[1:]  # flavor and deliciousness pairs
    
    from collections import defaultdict
    
    # Dictionary to store up to two largest deliciousness values per flavor
    top2 = defaultdict(list)
    
    # Parse input
    idx = 0
    for _ in range(N):
        f = int(FT[idx]); s = int(FT[idx+1])
        idx += 2
        top2[f].append(s)
    
    # For each flavor, keep only the two largest deliciousness values
    for f in top2:
        top2[f].sort(reverse=True)
        if len(top2[f]) > 2:
            top2[f] = top2[f][:2]
    
    # Prepare for scenario 1: different flavors
    # We'll collect the largest deliciousness per flavor in a list
    top1_list = []
    for f, arr in top2.items():
        if arr:  # at least one value
            top1_list.append((arr[0], f))
    
    # Sort descending by largest deliciousness
    top1_list.sort(key=lambda x: x[0], reverse=True)
    
    scenario1 = 0  # if we don't find at least 2 flavors, it will remain 0
    if len(top1_list) >= 2:
        # The top two in this list will be from distinct flavors by construction
        scenario1 = top1_list[0][0] + top1_list[1][0]
    
    # Scenario 2: same flavor
    scenario2 = 0
    for f, arr in top2.items():
        if len(arr) >= 2:
            s = arr[0]
            t = arr[1]
            # s is the larger one, t is the smaller one
            candidate = s + t//2
            if candidate > scenario2:
                scenario2 = candidate
    
    # Final answer
    print(max(scenario1, scenario2))

# Call main to execute the solution
main()