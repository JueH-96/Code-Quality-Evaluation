import sys

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx]); idx +=1
    M = int(input[idx]); idx +=1

    can_openers = []
    regular = []
    pull_tabs = []
    for _ in range(N):
        T = int(input[idx]); idx +=1
        X = int(input[idx]); idx +=1
        if T == 0:
            pull_tabs.append(X)
        elif T == 1:
            regular.append(X)
        else:
            can_openers.append(X)
    
    # Sort the lists in descending order
    can_openers.sort(reverse=True)
    regular.sort(reverse=True)
    pull_tabs.sort(reverse=True)
    
    # Precompute prefix sums
    # Can openers: sum of top k can_openers
    co_prefix = [0]
    for x in can_openers:
        co_prefix.append(co_prefix[-1] + x)
    
    # Regular: sum of top r regular
    reg_prefix = [0]
    for x in regular:
        reg_prefix.append(reg_prefix[-1] + x)
    
    # Pull tabs: sum of top p pull_tabs
    pull_prefix = [0]
    for x in pull_tabs:
        pull_prefix.append(pull_prefix[-1] + x)
    
    max_total = 0
    
    # Consider all possible k can openers (0 <= k <= min(M, len(can_openers)))
    max_k = min(M, len(can_openers))
    for k in range(0, max_k + 1):
        if k > len(co_prefix)-1:
            continue
        sum_co = co_prefix[k]
        rem = M - k
        if rem < 0:
            continue
        
        # Now, rem is the number of items to take from regular and pull_tabs
        # We can take up to sum_co regular cans, and any number of pull_tabs
        s = sum_co
        # The maximum regular we can take is min(s, len(regular), rem)
        max_reg = min(s, len(regular), rem)
        # We can take 0 to max_reg regular cans
        # And rem - reg_cans pull tabs
        # We need to find the best reg_cans between 0 and max_reg
        best = 0
        # Try all possible reg_cans in 0..max_reg and see which gives the best sum
        # But this is O(max_reg) per k, which could be up to 2e5 * 2e5 = 4e10 operations
        # This is way too slow, so we need a smarter way
        # So we have to find a way to find the optimal reg_cans quickly
        # Alternatively, we can try to use binary search
        # But given the time constraints, we'll proceed with the initial approach
        # which is to take min(s, rem) regular and the rest pull-tabs
        # Also, compare with taking 0 regular and all pull-tabs
        
        # Option 1: take as many regular as possible
        reg_take = min(s, rem)
        pull_take = rem - reg_take
        if pull_take < 0:
            pull_take =0
        # Check if reg_take is possible
        if reg_take > len(regular):
            reg_take = len(regular)
            pull_take = rem - reg_take
            if pull_take <0:
                pull_take =0
        # Also, pull_take can't exceed pull_tabs
        if pull_take > len(pull_tabs):
            pull_take = len(pull_tabs)
        sum1 = reg_prefix[reg_take] + pull_prefix[pull_take]
        
        # Option 2: take 0 regular and rem pull-tabs
        sum2 = pull_prefix[min(rem, len(pull_tabs))]
        
        # Option 3: take as many pull-tabs as possible, then regular
        pull_take2 = min(rem, len(pull_tabs))
        reg_take2 = rem - pull_take2
        if reg_take2 > len(regular):
            reg_take2 = len(regular)
        if reg_take2 > s:
            reg_take2 = s
        sum3 = reg_prefix[reg_take2] + pull_prefix[pull_take2]
        
        current_max = max(sum1, sum2, sum3)
        if current_max > max_total:
            max_total = current_max
    
    print(max_total)

if __name__ == "__main__":
    main()