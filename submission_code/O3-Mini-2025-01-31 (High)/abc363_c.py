def main():
    import sys, math
    sys.setrecursionlimit(10000)
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    K = int(data[1])
    S = data[2].strip()
    
    # A useful observation:
    # For any permutation of S to contain a palindrome of length K (with K>=2),
    # some letter must appear twice in consecutive mirror‐positions.
    # In particular, when S has all distinct letters,
    # no palindrome of length >=2 can appear (because a palindrome of even length requires two equal adjacent letters and for odd length the first and last must match).
    # Thus if S is all distinct, the answer is just the total number of permutations: N!.
    
    from collections import Counter
    freq = Counter(S)
    # Get a canonical order for our letters
    letters = sorted(freq.keys())
    
    if len(freq) == len(S):
        # All letters are distinct so no two adjacent letters (or any symmetric block) can form a palindrome.
        sys.stdout.write(str(math.factorial(N)))
        return

    # Otherwise, we have duplicates. In that case many intermediate “states” in the permutation tree may repeat.
    # We use a DP with memoization.
    #
    # The idea:
    #   We build the permutation character‐by‐character.
    #   We maintain a state defined by:
    #       • counts: a tuple of how many of each letter (in the sorted order "letters") remain to be placed.
    #       • tail: a tuple of the last min(K-1, current_length) letters of the building permutation.
    #   When we add a new letter, if len(tail) == K-1, then the contiguous block “tail+(new_letter,)”
    #   is of length K. We immediately check if that block is a palindrome. If it is, we prune that branch.
    #
    # Because K is at most 10 this check is very cheap.
    
    from functools import lru_cache
    
    @lru_cache(maxsize=None)
    def dp(counts, tail):
        rem = sum(counts)
        if rem == 0:
            return 1
        total = 0
        # Iterating over each available letter.
        for i in range(len(letters)):
            if counts[i] > 0:
                ch = letters[i]
                # If our tail already has K-1 letters, then appending ch will form a contiguous substring of length K.
                # We check if this new block is a palindrome.
                if len(tail) == K - 1:
                    new_block = tail + (ch,)
                    if new_block == new_block[::-1]:
                        continue  # Prune this branch.
                # Update the tail: we append ch and take the last K-1 letters.
                new_tail = (tail + (ch,))[-(K-1):] if K - 1 > 0 else ()
                # Update the counts by subtracting one copy of ch.
                new_counts = list(counts)
                new_counts[i] -= 1
                total += dp(tuple(new_counts), new_tail)
        return total

    init_counts = tuple(freq[ch] for ch in letters)
    answer = dp(init_counts, ())
    sys.stdout.write(str(answer))
    
if __name__ == '__main__':
    main()