def solve():
    import sys
    sys.setrecursionlimit(10**7)

    input_data = sys.stdin.read().strip().split()
    N, K = map(int, input_data[:2])
    S = input_data[2]

    from collections import Counter

    # Count the frequency of each distinct character in S
    freq_counter = Counter(S)
    distinct_chars = sorted(freq_counter.keys())
    freq_list = [freq_counter[ch] for ch in distinct_chars]
    d = len(distinct_chars)

    # Quickly check if a string is a palindrome
    def is_palindrome(st):
        return st == st[::-1]

    # Memo dictionary to cache results:
    # Key: (tuple_of_freq, suffix_string) -> Value: number_of_valid_permutations
    memo = {}

    # Recursively count permutations using DP + backtracking
    def dfs(freq, suffix):
        # If no characters remain, this is a complete valid permutation
        if all(f == 0 for f in freq):
            return 1

        state = (tuple(freq), suffix)
        if state in memo:
            return memo[state]

        total_ways = 0

        for i in range(d):
            if freq[i] > 0:
                ch = distinct_chars[i]
                # Create new suffix by appending this character
                new_suffix = suffix + ch

                # Check if the last K characters form a palindrome
                if len(new_suffix) >= K and is_palindrome(new_suffix[-K:]):
                    continue  # This would form an invalid palindrome substring

                # Use one instance of this character
                freq[i] -= 1

                # Keep only at most (K-1) last characters in suffix for next state
                if len(new_suffix) >= K:
                    next_suffix = new_suffix[-(K-1):]
                else:
                    next_suffix = new_suffix

                total_ways += dfs(freq, next_suffix)
                freq[i] += 1

        memo[state] = total_ways
        return total_ways

    # Compute and print result
    result = dfs(freq_list, "")
    print(result)

# Let's call solve() to execute the solution.
if __name__ == "__main__":
    solve()