import sys

# Function to read input faster
input = sys.stdin.readline

def solve():
    """
    Solves the problem of counting adjacent identical characters in substrings.
    Reads input, computes prefix sums, answers queries, and prints results.
    """
    n, q = map(int, input().split())
    s = input().strip()

    # prefix_sum[i] stores the number of indices p such that
    # 0 <= p < i-1 and s[p] == s[p+1].
    # This represents the count of adjacent identical pairs within the prefix s[0...i-1].
    # The array has size n+1, with indices 0 to n.
    prefix_sum = [0] * (n + 1)

    # Calculate prefix sums
    # The loop iterates from i = 2 up to n.
    # prefix_sum[i] is calculated based on prefix_sum[i-1] and the pair s[i-2], s[i-1].
    for i in range(2, n + 1):
        # Initialize current prefix sum with the previous one
        prefix_sum[i] = prefix_sum[i-1]
        # If the last two characters of the prefix s[0...i-1] are identical,
        # increment the count. These characters are at indices i-2 and i-1.
        if s[i-2] == s[i-1]:
            prefix_sum[i] += 1

    # Store results for all queries
    results = []
    for _ in range(q):
        # Read query bounds (1-based indexing)
        l, r = map(int, input().split())

        # Calculate the answer for the query [l, r].
        # The number of adjacent identical pairs in the substring s[l-1...r-1] (0-based)
        # corresponds to the number of indices p' such that l-1 <= p' <= r-2 and s[p'] == s[p'+1].
        # This count can be efficiently computed using the prefix sums:
        # count(l-1, r-2) = count(0, r-2) - count(0, l-2)
        # count(0, k) is the number of pairs ending at or before index k+1.
        # In terms of our prefix_sum array definition:
        # count(0, k) = prefix_sum[k+2]
        # So, count(l-1, r-2) = prefix_sum[(r-2)+2] - prefix_sum[(l-2)+2]
        #                  = prefix_sum[r] - prefix_sum[l]
        ans = prefix_sum[r] - prefix_sum[l]
        results.append(str(ans)) # Convert to string for joining later

    # Print all results, each on a new line
    print('
'.join(results))

# Execute the solve function
solve()