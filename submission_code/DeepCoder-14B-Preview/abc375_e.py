def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    people = []
    total_sum = 0
    for _ in range(N):
        A_i = int(input[idx])
        B_i = int(input[idx + 1])
        people.append((A_i, B_i))
        total_sum += B_i
        idx += 2

    if total_sum % 3 != 0:
        print(-1)
        return

    target = total_sum // 3

    # Initialize DP: key is (sum1, sum2, sum3), value is the minimal number of changes
    dp = {(0, 0, 0): 0}

    for a_i, b_i in people:
        new_dp = {}
        for (sum1, sum2, sum3) in dp:
            current_changes = dp[(sum1, sum2, sum3)]
            
            # Assign to team 1
            new_sum1 = sum1 + b_i
            if new_sum1 <= target:
                new_changes = current_changes + (1 if a_i != 1 else 0)
                key = (new_sum1, sum2, sum3)
                if key in new_dp:
                    if new_changes < new_dp[key]:
                        new_dp[key] = new_changes
                else:
                    new_dp[key] = new_changes
            
            # Assign to team 2
            new_sum2 = sum2 + b_i
            if new_sum2 <= target:
                new_changes = current_changes + (1 if a_i != 2 else 0)
                key = (sum1, new_sum2, sum3)
                if key in new_dp:
                    if new_changes < new_dp[key]:
                        new_dp[key] = new_changes
                else:
                    new_dp[key] = new_changes
            
            # Assign to team 3
            new_sum3 = sum3 + b_i
            if new_sum3 <= target:
                new_changes = current_changes + (1 if a_i != 3 else 0)
                key = (sum1, sum2, new_sum3)
                if key in new_dp:
                    if new_changes < new_dp[key]:
                        new_dp[key] = new_changes
                else:
                    new_dp[key] = new_changes
        
        # Update dp to new_dp
        dp = new_dp

    min_changes = float('inf')
    for (sum1, sum2, sum3), changes in dp.items():
        if sum1 == sum2 == sum3:
            if changes < min_changes:
                min_changes = changes

    if min_changes != float('inf'):
        print(min_changes)
    else:
        print(-1)

if __name__ == '__main__':
    main()