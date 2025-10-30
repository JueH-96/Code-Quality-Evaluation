MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:2+N]))
    
    # Precompute prefix sums
    prefix = [0] * (N + 1)
    for i in range(N):
        prefix[i+1] = prefix[i] + A[i]
    
    # Use a dictionary to store the frequency of prefix sums
    from collections import defaultdict
    freq = defaultdict(int)
    freq[0] = 1
    result = 1
    for i in range(1, N+1):
        # Calculate the required sum to avoid K
        required = prefix[i] - K
        # If required sum exists in freq, it means there is a subsequence summing to K
        if required in freq:
            result = 0
            break
        # Update the frequency of the current prefix sum
        freq[prefix[i]] += 1
    
    if result == 0:
        print(0)
    else:
        # The number of ways is 2^(number of possible cuts)
        # Since no subsequence sums to K, all cuts are allowed
        # The number of cuts is N-1, but since we have to avoid any subsequence summing to K
        # and we have already ensured that, the total ways is 2^(N-1)
        # However, in the case where no subsequence sums to K, the total ways is 2^(N-1)
        # But in the case where some subsequence sums to K, the result is 0
        # So, since we have already checked that no subsequence sums to K, the result is 2^(N-1)
        print(pow(2, N-1, MOD))

if __name__ == "__main__":
    main()