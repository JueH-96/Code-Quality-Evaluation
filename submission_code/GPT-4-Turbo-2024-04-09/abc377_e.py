def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    P = list(map(int, data[2:]))
    
    # Convert P from 1-based index to 0-based index for easier manipulation
    P = [x - 1 for x in P]
    
    # To handle large K efficiently, we use exponentiation by squaring on the permutation
    # We need to find P^k, where P is a permutation and k is a large number
    
    # Function to compute the permutation to the power of exp
    def permute_power(perm, exp):
        n = len(perm)
        result = list(range(n))  # Start with identity permutation
        base = perm[:]
        
        while exp > 0:
            if exp % 2 == 1:
                result = apply_permutation(base, result)
            base = apply_permutation(base, base)
            exp //= 2
        
        return result
    
    # Function to apply permutation perm to array arr
    def apply_permutation(perm, arr):
        return [arr[perm[i]] for i in range(len(arr))]
    
    # Calculate P^K
    final_permutation = permute_power(P, K)
    
    # Convert result back to 1-based index
    final_permutation = [x + 1 for x in final_permutation]
    
    # Print the result
    print(" ".join(map(str, final_permutation)))

if __name__ == "__main__":
    main()