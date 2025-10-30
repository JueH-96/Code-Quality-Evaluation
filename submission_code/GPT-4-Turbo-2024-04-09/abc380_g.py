MOD = 998244353

def modinv(x, p):
    """ Return modular inverse of x under modulo p """
    return pow(x, p - 2, p)

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    P = list(map(int, data[2:2+N]))
    
    # Calculate initial number of inversions
    initial_inversions = 0
    for i in range(N):
        for j in range(i + 1, N):
            if P[i] > P[j]:
                initial_inversions += 1
    
    if K == 1:
        # If K is 1, no change in permutation after shuffle
        print(initial_inversions % MOD)
        return
    
    # Expected inversions in a K-length segment when fully shuffled
    expected_inversions_in_K = K * (K - 1) // 2 // 2
    
    # Calculate the expected number of inversions after the operation
    expected_inversions = initial_inversions
    
    # We need to adjust the inversions for each possible segment [i, i+K-1]
    for start in range(N - K + 1):
        # Calculate the number of inversions in the current segment [start, start+K-1]
        current_segment_inversions = 0
        for i in range(start, start + K):
            for j in range(i + 1, start + K):
                if P[i] > P[j]:
                    current_segment_inversions += 1
        
        # We replace the inversions in this segment with the expected number of inversions
        # after a random shuffle of this segment
        expected_inversions += (expected_inversions_in_K - current_segment_inversions)
    
    # The expected value of inversions after one random operation
    # Since each segment can be chosen with probability 1 / (N - K + 1)
    total_possible_starts = N - K + 1
    expected_value = expected_inversions * modinv(total_possible_starts, MOD) % MOD
    
    print(expected_value)

if __name__ == "__main__":
    main()