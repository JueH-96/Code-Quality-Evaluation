import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(10000)
    
    # Read input
    data = sys.stdin.read().strip().split()
    N, K = map(int, data[:2])
    S = data[2]
    
    # Frequency array for 'a' to 'z'
    freq = [0] * 26
    for ch in S:
        freq[ord(ch) - ord('a')] += 1
    
    count = 0  # will hold the final count
    
    # We build the permutation step by step, pruning as soon as a K-length palindrome appears.
    perm = [None] * N  # placeholder for current building permutation
    
    def dfs(pos):
        nonlocal count
        # If we've built a full permutation, it's valid
        if pos == N:
            count += 1
            return
        
        # Try placing each letter that remains
        for c in range(26):
            if freq[c] == 0:
                continue
            # place letter c at position pos
            freq[c] -= 1
            perm[pos] = c
            
            # If we have at least K letters, check the last K for a palindrome
            ok = True
            if pos + 1 >= K:
                i = pos
                j = pos + 1 - K
                # check substring perm[j..i] of length K
                # is it a palindrome?
                for t in range(K // 2):
                    if perm[j + t] != perm[i - t]:
                        break
                else:
                    # if we did not break, it's a palindrome --> prune
                    ok = False
            
            if ok:
                dfs(pos + 1)
            
            # backtrack
            freq[c] += 1
    
    dfs(0)
    
    # Output the result
    print(count)

if __name__ == "__main__":
    main()