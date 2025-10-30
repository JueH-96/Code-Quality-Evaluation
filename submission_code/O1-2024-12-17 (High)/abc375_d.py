def main():
    import sys
    S = sys.stdin.readline().strip()

    # Lists to hold positions (1-based) of each uppercase letter
    pos = [[] for _ in range(26)]
    for i, ch in enumerate(S):
        pos[ord(ch) - ord('A')].append(i+1)

    def sum_of_pairs(positions):
        """Return the sum over all pairs (i, k) in 'positions' of (k - i - 1)."""
        m = len(positions)
        if m < 2:
            return 0
        prefix_sum = [0] * (m + 1)
        # Build prefix sums for positions
        for i in range(m):
            prefix_sum[i+1] = prefix_sum[i] + positions[i]
        total = 0
        # Sum up (k - i) for all pairs, then we'll subtract the number of pairs
        for y in range(2, m + 1):
            pY = positions[y - 1]
            total += pY * (y - 1) - prefix_sum[y - 1]
        # Subtract one per pair to get (k - i - 1)
        total -= (m * (m - 1)) // 2
        return total

    # Sum results for each letter
    ans = 0
    for i in range(26):
        ans += sum_of_pairs(pos[i])

    print(ans)

# Do not forget to call the main() function
main()