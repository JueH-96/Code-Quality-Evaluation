# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

# dp[i][j] = maximum score from subarray [i, j]
dp = {}

def solve(l, r):
    if (l, r) in dp:
        return dp[(l, r)]
    
    length = r - l + 1
    
    # Base cases
    if length <= 1:
        return 0
    if length == 2:
        return abs(a[l] - a[r])
    
    result = 0
    
    # Try all possible ways to pair elements
    # For odd length, one element will remain unpaired
    if length % 2 == 0:
        # All elements must be paired
        # Try pairing first element with each other element
        for k in range(l + 1, r + 1, 2):
            # Pair a[l] with a[k]
            score = abs(a[l] - a[k])
            # Recursively solve remaining segments
            left_score = solve(l + 1, k - 1) if k > l + 1 else 0
            right_score = solve(k + 1, r) if k < r else 0
            result = max(result, score + left_score + right_score)
    else:
        # One element will remain unpaired
        # Try leaving each element unpaired
        for skip in range(l, r + 1):
            score = 0
            if skip == l:
                score = solve(l + 1, r)
            elif skip == r:
                score = solve(l, r - 1)
            else:
                # Elements before skip must have even count
                # Elements after skip must have even count
                if (skip - l) % 2 == 0 and (r - skip) % 2 == 0:
                    score = solve(l, skip - 1) + solve(skip + 1, r)
            result = max(result, score)
    
    dp[(l, r)] = result
    return result

print(solve(0, n - 1))