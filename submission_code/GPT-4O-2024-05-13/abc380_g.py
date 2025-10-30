# YOUR CODE HERE
import sys
input = sys.stdin.read

MOD = 998244353

def mod_inv(a, mod):
    return pow(a, mod - 2, mod)

def main():
    data = input().strip().split()
    N = int(data[0])
    K = int(data[1])
    P = list(map(int, data[2:]))
    
    # Number of possible starting indices
    num_intervals = N - K + 1
    
    # Expected number of inversions in a shuffled subarray of length K
    expected_inversions_in_subarray = K * (K - 1) // 4
    
    # Total expected inversions
    total_expected_inversions = expected_inversions_in_subarray * num_intervals
    
    # Calculate the final expected value modulo 998244353
    result = (total_expected_inversions * mod_inv(num_intervals, MOD)) % MOD
    
    print(result)

if __name__ == "__main__":
    main()