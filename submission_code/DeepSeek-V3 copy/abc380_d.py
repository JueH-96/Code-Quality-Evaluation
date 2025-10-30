# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    S = data[0]
    Q = int(data[1])
    K_list = list(map(int, data[2:2+Q]))
    
    n = len(S)
    # The length of S after k operations is n * 2^k
    # Since 2^k grows very fast, for k >= 60, the length is larger than 10^18
    # So for K_i up to 10^18, we can consider the pattern after 60 operations
    
    # The pattern after 60 operations is the same as after 1 operation, but repeated
    # So the length is n * 2^60, which is larger than 10^18
    
    # To find the K_i-th character, we can find the position in the original S or its complement
    # The pattern is S, T, S, T, ... where T is the complement of S
    
    # The length of S after k operations is n * 2^k
    # For K_i, we can find the smallest k such that n * 2^k >= K_i
    # Then, the K_i-th character is in the k-th level
    
    # But since 2^k grows exponentially, for K_i up to 10^18, k is at most 60
    
    # So for each K_i, we can find the position in the original S or its complement
    
    # The pattern is S, T, S, T, ... and so on
    
    # The K_i-th character is:
    # if K_i <= n: S[K_i-1]
    # else: find the level k where n * 2^{k-1} < K_i <= n * 2^k
    # then, the character is in the k-th level, which is either S or T
    
    # To find the level k, we can use binary search
    
    # Precompute the lengths after each operation
    # lengths[k] = n * 2^k
    # Since 2^60 is about 1e18, we can precompute up to 60
    
    max_k = 60
    lengths = [n * (1 << k) for k in range(max_k + 1)]
    
    result = []
    for K in K_list:
        if K <= n:
            result.append(S[K-1])
            continue
        # Find the smallest k such that lengths[k] >= K
        low = 0
        high = max_k
        k = 0
        while low <= high:
            mid = (low + high) // 2
            if lengths[mid] >= K:
                k = mid
                high = mid - 1
            else:
                low = mid + 1
        # Now, the K-th character is in the k-th level
        # The k-th level is S if k is even, T if k is odd
        # The position in the original S is (K - 1) % n
        pos = (K - 1) % n
        if k % 2 == 0:
            result.append(S[pos])
        else:
            # T is the complement of S
            # To get T, swap case of S
            T = S.swapcase()
            result.append(T[pos])
    
    print(' '.join(result))

if __name__ == "__main__":
    main()