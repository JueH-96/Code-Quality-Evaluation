# YOUR CODE HERE
import sys
from collections import Counter

def main():
    import sys
    sys.setrecursionlimit(10000)
    N, K = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()
    counter = Counter(S)
    unique_chars = sorted(counter.keys())
    counts = [counter[ch] for ch in unique_chars]
    total = 0
    num_unique = len(unique_chars)

    def backtrack(path_length, counts, last_k_minus_1):
        nonlocal total
        if path_length == N:
            total +=1
            return
        prev = None
        for i in range(num_unique):
            if counts[i] ==0:
                continue
            if unique_chars[i] == prev:
                continue
            # Choose unique_chars[i]
            # Check for duplicates
            # To skip duplicates, ensure that if the previous character is same and not used, skip
            # Since we iterate in order and skip same as previous, it's handled
            # Check if adding this character creates a K-length palindrome
            if K <= path_length +1:
                # Build the window of last K-1 characters plus current
                window = last_k_minus_1 + (unique_chars[i], )
                if len(window) > K:
                    window = window[-K:]
                if len(window) == K and ''.join(window) == ''.join(window)[::-1]:
                    continue
            # Proceed to choose this character
            counts[i] -=1
            if K >1:
                if len(last_k_minus_1) < K-1:
                    new_last = last_k_minus_1 + (unique_chars[i], )
                else:
                    new_last = last_k_minus_1[1:] + (unique_chars[i], )
            else:
                new_last = ()
            backtrack(path_length +1, counts, new_last)
            counts[i] +=1
            prev = unique_chars[i]

    backtrack(0, counts, ())
    print(total)

if __name__ == "__main__":
    main()