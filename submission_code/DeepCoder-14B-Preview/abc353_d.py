MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    if N < 2:
        print(0)
        return
    
    # Compute the number of digits for each element
    d = [len(str(x)) for x in A]
    
    # Compute prefix sums S where S[k] is sum of A[0..k-1]
    S = [0] * (N + 1)
    for k in range(1, N + 1):
        S[k] = S[k-1] + A[k-1]
    
    # Calculate sum_part1
    sum_part1 = 0
    for j in range(1, N):  # j is zero-based from 1 to N-1
        pow10 = pow(10, d[j], MOD)
        s_mod = S[j] % MOD
        term = (s_mod * pow10) % MOD
        sum_part1 = (sum_part1 + term) % MOD
    
    # Calculate sum_part2
    sum_part2 = 0
    for j in range(N):
        term = (j * A[j]) % MOD
        sum_part2 = (sum_part2 + term) % MOD
    
    # Total sum
    total = (sum_part1 + sum_part2) % MOD
    print(total)

if __name__ == "__main__":
    main()