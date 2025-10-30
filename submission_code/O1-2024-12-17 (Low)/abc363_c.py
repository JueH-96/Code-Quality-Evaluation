def main():
    import sys
    sys.setrecursionlimit(10**7)
    
    # Read inputs
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    K = int(data[1])
    S = data[2]
    
    # Count the frequency of each character in S
    from collections import Counter
    freq = Counter(S)
    
    # Function to check if a string of length K is a palindrome
    # (Since K <= 10, a simple check is efficient enough)
    def is_k_palindrome(substring):
        return substring == substring[::-1]
    
    # We'll use backtracking with pruning.
    # We keep a current partial permutation (as a list of chars for efficiency),
    # and if its last K characters form a palindrome, we prune that branch.
    result = 0
    
    def backtrack(partial_length):
        nonlocal result
        
        # If we've placed N characters without forming a K-length palindrome substring,
        # that means we've built a valid permutation.
        if partial_length == N:
            result += 1
            return
        
        # Try each character that still has a positive frequency
        for ch in freq:
            if freq[ch] > 0:
                # Place this character
                freq[ch] -= 1
                partial.append(ch)
                
                # Check if the last K characters form a palindrome
                if partial_length + 1 >= K:
                    # Only check the substring of length K
                    if not is_k_palindrome(partial[-K:]):
                        backtrack(partial_length + 1)
                else:
                    # Not enough length to form a K-length palindrome yet
                    backtrack(partial_length + 1)
                
                # Backtrack
                partial.pop()
                freq[ch] += 1
    
    partial = []
    backtrack(0)
    
    print(result)

# Do not forget to call main()
if __name__ == "__main__":
    main()