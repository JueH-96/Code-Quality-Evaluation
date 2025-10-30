def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    X = int(input[idx])
    idx += 1
    Y = int(input[idx])
    idx += 1
    dishes = []
    for _ in range(N):
        A = int(input[idx])
        idx += 1
        B = int(input[idx])
        idx += 1
        dishes.append((A, B))
    
    # Initialize DP tables
    valid_dp = [{} for _ in range(N + 2)]  # valid_dp[m] is a dict of sum_A to minimal sum_B
    valid_dp[0][0] = 0
    invalid_dp = [False] * (N + 2)
    
    for A, B in dishes:
        # Iterate from N down to 0 to avoid reusing the same dish
        for m in reversed(range(N + 1)):
            current = valid_dp[m]
            # Iterate over a copy of keys to avoid runtime errors from dict changes
            for a_prev in list(current.keys()):
                sum_b_prev = current[a_prev]
                new_a = a_prev + A
                new_b = sum_b_prev + B
                # Check if the new sums are within limits
                if new_a <= X and new_b <= Y:
                    # Update valid_dp[m+1] with the minimal sum_b for new_a
                    if new_a in valid_dp[m + 1]:
                        if new_b < valid_dp[m + 1][new_a]:
                            valid_dp[m + 1][new_a] = new_b
                    else:
                        valid_dp[m + 1][new_a] = new_b
                else:
                    # If the sum exceeds, mark invalid_dp[m+1]
                    if not invalid_dp[m + 1]:
                        invalid_dp[m + 1] = True
    
    # Determine the maximum number of dishes
    max_dishes = 0
    for m in reversed(range(1, N + 1)):
        if valid_dp[m] or invalid_dp[m]:
            max_dishes = m
            break
    print(max_dishes)

if __name__ == "__main__":
    main()