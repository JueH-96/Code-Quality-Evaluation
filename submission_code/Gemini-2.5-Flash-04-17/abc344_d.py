import sys

# Read target string T
T = sys.stdin.readline().strip()
len_T = len(T)

# Read N
N = int(sys.stdin.readline())

# Read bags of strings
bags = []
for _ in range(N):
    line = sys.stdin.readline().split()
    # The first element is A_i (number of strings), followed by the strings.
    # We don't explicitly need A_i, just the list of strings.
    strings_in_bag = line[1:]
    bags.append(strings_in_bag)

# Dynamic programming
# dp[j] will store the minimum cost to form the prefix T[:j] using bags processed so far.
# Initialize dp with a value representing infinity.
# The maximum possible cost is N (if we pick one string from each of the N bags).
# So N + 1 is a safe infinity value, as cost cannot exceed N.
infinity = N + 1
dp = [infinity] * (len_T + 1)

# Base case: Cost to form an empty prefix is 0.
dp[0] = 0

# Iterate through each bag, processing them in order from 1 to N.
# The outer loop simulates processing bag i.
for bag in bags:
    # For each string in the current bag, we consider using it to extend a previously
    # formed prefix.
    for s in bag:
        len_s = len(s)
        # We iterate through possible ending positions (j) in the target string T.
        # If we use string s from the current bag to form T[:j], then T[j-len_s:j]
        # must be equal to s, and we must have previously formed T[:j-len_s].
        # We iterate j from len_T down to len_s to ensure that when considering
        # reaching T[:j] by appending s, the value dp[j-len_s] corresponds to the
        # minimum cost *before* potentially using a string from the current bag
        # to reach T[:j-len_s] itself (if s was a suffix of a prefix ending at j-len_s).
        # The range is inclusive of len_T and len_s. The step is -1.
        for j in range(len_T, len_s - 1, -1):
            # Check if string s is a suffix of the prefix T[:j].
            # T[j - len_s : j] gets the substring of T from index j - len_s up to (but not including) j.
            if T[j - len_s : j] == s:
                # If the prefix T[:j-len_s] was reachable (cost is not infinity)
                if dp[j - len_s] != infinity:
                    # Update the minimum cost to reach T[:j] by considering
                    # the cost of reaching T[:j-len_s] and adding the cost of
                    # using string s (which is 1).
                    dp[j] = min(dp[j], dp[j - len_s] + 1)

# After processing all bags, the minimum cost to form the entire string T
# is stored in dp[len_T].
result = dp[len_T]

# If the result is still infinity, it means it's impossible to form the string T.
if result == infinity:
    print(-1)
else:
    print(result)