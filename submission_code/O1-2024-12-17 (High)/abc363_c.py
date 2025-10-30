def main():
    import sys
    from collections import Counter
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    K = int(data[1])
    S = data[2]
    
    # Count frequencies of each character in S
    freq = Counter(S)
    unique_chars = list(freq.keys())
    
    # We'll build permutations in partial_arr of length N
    partial_arr = [None] * N
    
    # Use a list to store answer so we can modify inside nested function
    ans = [0]
    
    def backtrack(idx):
        # If we've placed all characters, we have a valid permutation
        if idx == N:
            ans[0] += 1
            return
        
        # Try placing each character that still has available frequency
        for c in unique_chars:
            if freq[c] > 0:
                partial_arr[idx] = c
                freq[c] -= 1
                
                # Check if the last K characters form a palindrome
                found_pal = False
                if idx + 1 >= K:
                    check = True
                    start = idx - K + 1
                    # Verify if substring partial_arr[start..idx] is a palindrome
                    for i in range(K // 2):
                        if partial_arr[start + i] != partial_arr[idx - i]:
                            check = False
                            break
                    if check:
                        found_pal = True
                
                # If we did NOT form a palindrome of length K, recurse
                if not found_pal:
                    backtrack(idx + 1)
                
                # Revert changes
                freq[c] += 1
    
    # Start backtracking from index 0
    backtrack(0)
    
    # Print the result
    print(ans[0])

# Do not forget to call main()
if __name__ == "__main__":
    main()