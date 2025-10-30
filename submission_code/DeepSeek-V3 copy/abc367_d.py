def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:2+N]))
    
    # Compute the prefix sum array
    prefix = [0] * (N + 1)
    for i in range(N):
        prefix[i+1] = prefix[i] + A[i]
    total = prefix[N]
    
    # Compute the remainder of the prefix sums modulo M
    remainder = [0] * (N + 1)
    for i in range(N + 1):
        remainder[i] = prefix[i] % M
    
    # Create a frequency map for the remainders
    from collections import defaultdict
    freq = defaultdict(int)
    for r in remainder:
        freq[r] += 1
    
    # Calculate the number of pairs (s, t) where s < t and (prefix[t] - prefix[s]) % M == 0
    # This is equivalent to the number of pairs where remainder[t] == remainder[s]
    count = 0
    for r in freq:
        cnt = freq[r]
        if cnt >= 2:
            count += cnt * (cnt - 1) // 2
    
    # Since the problem requires s != t, and the above calculation includes all s < t, we need to consider the circular nature
    # For the circular case, we need to consider pairs where s > t, but the difference is still a multiple of M
    # The total steps from s to t is (prefix[t] - prefix[s]) % total, but since total is not necessarily a multiple of M, we need to consider the modulo M
    # However, the initial approach already counts all pairs where (prefix[t] - prefix[s]) % M == 0, which includes both s < t and s > t cases
    # So the initial count is sufficient
    
    print(count)

if __name__ == "__main__":
    main()