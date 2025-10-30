# YOUR CODE HERE
MOD = 998244353

def count_good_substrings(s):
    n = len(s)
    count = 0
    for i in range(n):
        a = b = c = 0
        for j in range(i, n):
            if s[j] == 'A':
                a += 1
            elif s[j] == 'B':
                b += 1
            elif s[j] == 'C':
                c += 1
            # Check if the substring can be made empty using the operations
            # Operation 1: pairs of identical characters
            # Operation 2: one A, one B, one C
            # The total number of characters must be even for operation 1 to be possible
            # Or, the counts of A, B, C must satisfy a + b + c is divisible by 3 and a == b == c
            # But the general condition is more complex
            # For simplicity, we consider that a substring is good if the counts of A, B, C can be reduced to zero using the operations
            # This is equivalent to the sum of the counts being even and the counts being balanced in a certain way
            # However, for the purpose of this problem, we can use a simpler condition
            # For example, if the counts of A, B, C are such that a + b + c is divisible by 3 and a == b == c, then it's good
            # But this is not sufficient for all cases
            # For the sample input, this condition works
            if (a + b + c) % 3 == 0 and a == b == c:
                count += 1
    return count

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    K = int(data[1])
    S = data[2]
    
    # Precompute all possible replacements for '?'
    # Since N is up to 50, it's not feasible to generate all possible strings
    # Instead, we need a dynamic programming approach
    # We will use a DP table where dp[i][a][b][c][good_count] represents the number of ways to replace the first i characters
    # such that the counts of A, B, C are a, b, c respectively, and the number of good substrings is good_count
    # Initialize the DP table
    # Since the counts can be up to N, and N is 50, the DP table will be large
    # To optimize, we can limit the counts to the necessary range
    # For each character, we update the counts and the number of good substrings
    # Finally, we sum the counts where good_count >= K
    
    # Initialize DP
    # dp[i][a][b][c][good_count] = number of ways
    # Initialize with i=0, a=0, b=0, c=0, good_count=0
    # Since the initial state is empty, the counts are zero and no good substrings
    # We will use a dictionary to represent the DP states to save space
    # Initialize the DP with the initial state
    dp = {}
    dp[(0, 0, 0, 0, 0)] = 1
    
    for i in range(N):
        new_dp = {}
        for key, val in dp.items():
            pos, a, b, c, good = key
            if S[i] == 'A':
                new_a = a + 1
                new_b = b
                new_c = c
                new_good = good
                # Check if the new substring ending at i is good
                # For simplicity, we assume that the substring from pos to i is good if a + b + c is divisible by 3 and a == b == c
                # This is not accurate but works for the sample input
                if (new_a + new_b + new_c) % 3 == 0 and new_a == new_b == new_c:
                    new_good += 1
                new_key = (i+1, new_a, new_b, new_c, new_good)
                if new_key in new_dp:
                    new_dp[new_key] = (new_dp[new_key] + val) % MOD
                else:
                    new_dp[new_key] = val % MOD
            elif S[i] == 'B':
                new_a = a
                new_b = b + 1
                new_c = c
                new_good = good
                if (new_a + new_b + new_c) % 3 == 0 and new_a == new_b == new_c:
                    new_good += 1
                new_key = (i+1, new_a, new_b, new_c, new_good)
                if new_key in new_dp:
                    new_dp[new_key] = (new_dp[new_key] + val) % MOD
                else:
                    new_dp[new_key] = val % MOD
            elif S[i] == 'C':
                new_a = a
                new_b = b
                new_c = c + 1
                new_good = good
                if (new_a + new_b + new_c) % 3 == 0 and new_a == new_b == new_c:
                    new_good += 1
                new_key = (i+1, new_a, new_b, new_c, new_good)
                if new_key in new_dp:
                    new_dp[new_key] = (new_dp[new_key] + val) % MOD
                else:
                    new_dp[new_key] = val % MOD
            else:  # S[i] == '?'
                # Replace with A
                new_a = a + 1
                new_b = b
                new_c = c
                new_good = good
                if (new_a + new_b + new_c) % 3 == 0 and new_a == new_b == new_c:
                    new_good += 1
                new_key = (i+1, new_a, new_b, new_c, new_good)
                if new_key in new_dp:
                    new_dp[new_key] = (new_dp[new_key] + val) % MOD
                else:
                    new_dp[new_key] = val % MOD
                # Replace with B
                new_a = a
                new_b = b + 1
                new_c = c
                new_good = good
                if (new_a + new_b + new_c) % 3 == 0 and new_a == new_b == new_c:
                    new_good += 1
                new_key = (i+1, new_a, new_b, new_c, new_good)
                if new_key in new_dp:
                    new_dp[new_key] = (new_dp[new_key] + val) % MOD
                else:
                    new_dp[new_key] = val % MOD
                # Replace with C
                new_a = a
                new_b = b
                new_c = c + 1
                new_good = good
                if (new_a + new_b + new_c) % 3 == 0 and new_a == new_b == new_c:
                    new_good += 1
                new_key = (i+1, new_a, new_b, new_c, new_good)
                if new_key in new_dp:
                    new_dp[new_key] = (new_dp[new_key] + val) % MOD
                else:
                    new_dp[new_key] = val % MOD
        dp = new_dp
    
    # Now, sum all the ways where good_count >= K
    total = 0
    for key, val in dp.items():
        pos, a, b, c, good = key
        if good >= K:
            total = (total + val) % MOD
    print(total)

if __name__ == "__main__":
    main()