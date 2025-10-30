# Read the target string T
T = input().strip()

# Read the number of bags
N = int(input())

# Read each bag's contents
bags = []
for _ in range(N):
    parts = input().split()
    a_i = parts[0]
    strings = parts[1:]
    bags.append(strings)

# Initialize the DP dictionary with the starting state
dp = {"": 0}

# Process each bag
for bag in bags:
    new_dp = {}
    for current_s in dp:
        current_cost = dp[current_s]
        
        # Option 1: Do nothing
        if current_s in new_dp:
            if current_cost < new_dp[current_s]:
                new_dp[current_s] = current_cost
        else:
            new_dp[current_s] = current_cost
        
        # Option 2: Add each string in the current bag
        for s in bag:
            new_s = current_s + s
            new_cost = current_cost + 1
            if new_s in new_dp:
                if new_cost < new_dp[new_s]:
                    new_dp[new_s] = new_cost
            else:
                new_dp[new_s] = new_cost
    dp = new_dp

# Check if T is reachable and output the result
if T in dp:
    print(dp[T])
else:
    print(-1)