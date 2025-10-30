def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    # We'll read pairs (F_i, S_i).
    # We want to compute:
    # 1) The best sum of two cups with different flavors: s + t (s >= t).
    # 2) The best sum of two cups with the same flavor: s + t/2 (s >= t).
    # Then take the maximum of those.
    
    # Step 1: For each flavor, track the top two deliciousness values.
    # Step 2: Determine the maximum possible sum for the "same flavor" case.
    # Step 3: Determine the maximum possible sum for the "different flavors" case
    #         by taking the top flavor and the next best distinct flavor.
    # Step 4: Print the maximum of these two cases.
    
    # Dictionary: flavor -> list of top 2 deliciousness values
    flavor_dict = {}
    idx = 1
    for _ in range(N):
        f = int(input_data[idx]); s = int(input_data[idx+1])
        idx += 2
        if f not in flavor_dict:
            flavor_dict[f] = []
        flavor_dict[f].append(s)
    
    # For each flavor, keep only the top 2 deliciousness values
    # and compute the best same-flavor satisfaction if possible
    max_same_flavor = 0  # keep track of highest s + t/2
    flavor_best = []     # list of (best_s, flavor) for different-flavor scenario
    for f, arr in flavor_dict.items():
        arr.sort(reverse=True)
        # Keep track of the best one for different-flavor scenario
        flavor_best.append((arr[0], f))
        # If there's at least 2 for same flavor, update max_same_flavor
        if len(arr) >= 2:
            # s >= t by sorting
            s, t = arr[0], arr[1]
            # satisfaction = s + t/2
            val = s + t // 2
            if val > max_same_flavor:
                max_same_flavor = val
    
    # Now compute the best sum from two distinct flavors
    # Sort flavor_best by best_s descending
    flavor_best.sort(key=lambda x: x[0], reverse=True)
    max_diff_flavor = 0
    if len(flavor_best) >= 2:
        max_diff_flavor = flavor_best[0][0] + flavor_best[1][0]
    
    # The answer is the maximum of these two cases
    answer = max(max_same_flavor, max_diff_flavor)
    print(answer)

def main():
    solve()

if __name__ == "__main__":
    main()