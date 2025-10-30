def main():
    import sys
    sys.setrecursionlimit(10**7)
    from collections import Counter

    # Read inputs
    N, K = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()

    # Count the frequency of each character
    freq = Counter(S)
    counts = [0]*26
    for ch, c in freq.items():
        counts[ord(ch) - ord('a')] = c

    # Memo dictionary to cache results:
    # key = (tuple_of_counts, last_chars_string) -> number of valid permutations
    memo = {}

    def backtrack(counts, last_chars):
        # If all characters have been used, that's 1 valid permutation
        if sum(counts) == 0:
            return 1

        key = (tuple(counts), last_chars)
        if key in memo:
            return memo[key]

        total = 0
        # Try placing each character that is still available
        for i in range(26):
            if counts[i] > 0:
                c = chr(i + ord('a'))
                new_last = last_chars + c
                # Check if adding this character created a palindrome of length K
                if len(new_last) >= K:
                    segment = new_last[-K:]
                    if segment == segment[::-1]:
                        # This would create a forbidden palindrome substring
                        continue
                
                counts[i] -= 1
                # Keep only the last (K-1) characters in new_last for future checks
                if len(new_last) >= K - 1:
                    next_last = new_last[-(K - 1):]
                else:
                    next_last = new_last

                total += backtrack(counts, next_last)
                counts[i] += 1

        memo[key] = total
        return total

    ans = backtrack(counts, "")
    print(ans)

# Remember to call main()!
if __name__ == "__main__":
    main()