def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    it = iter(input_data)
    N = int(next(it))
    X = int(next(it))
    
    # Separate foods into three groups by vitamin type.
    foods_by_vit = [ [] for _ in range(3) ]
    for _ in range(N):
        v = int(next(it))
        a = int(next(it))
        c = int(next(it))
        foods_by_vit[v-1].append((a, c))
        
    # For each vitamin group, we build a DP array:
    # dp[c] = maximum vitamin units obtainable using exactly c calories.
    # Only calories up to X matter.
    def compute_dp(food_list):
        dp = [-1] * (X+1)
        dp[0] = 0
        for a, c in food_list:
            # Process in reverse order to ensure one cannot choose the same food twice.
            for cal in range(X - c, -1, -1):
                if dp[cal] != -1:
                    new_val = dp[cal] + a
                    if new_val > dp[cal + c]:
                        dp[cal + c] = new_val
        return dp

    dp_list = []
    max_vit_each = []
    for i in range(3):
        dp = compute_dp(foods_by_vit[i])
        dp_list.append(dp)
        # The maximum obtainble vitamin for group i is the max value in dp.
        max_vit_each.append(max(dp))
    
    # The overall best possible minimum vitamin intake is bounded by the smallest maximum 
    # we can obtain among the three vitamins.
    r_max = min(max_vit_each)
    
    # We now binary search for the maximum integer r (minimum vitamin intake among vitamins) 
    # that can be achieved with the total calorie cost of selected items <= X.
    # For a given r, in each vitamin group, we need to find the minimal calorie cost required 
    # to get at least r vitamin units.
    def feasible(r):
        total_cal = 0
        for dp in dp_list:
            min_cal = None
            # Check each calorie cost from 0 to X and find the smallest cost that gives at least r units.
            for cal in range(X+1):
                if dp[cal] >= r:
                    min_cal = cal
                    break
            if min_cal is None:
                return False
            total_cal += min_cal
            if total_cal > X:
                return False
        return True

    lo, hi = 0, r_max
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if feasible(mid):
            lo = mid
        else:
            hi = mid - 1
    
    print(lo)

if __name__ == '__main__':
    main()