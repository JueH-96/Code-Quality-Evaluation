import sys

MOD = 998244353

def main():
    n = int(sys.stdin.readline())
    a = list(map(int, sys.stdin.readline().split()))
    
    # Precompute pow10 for each element
    pow10_list = []
    for num in a:
        s = str(num)
        d = len(s)
        p = pow(10, d, MOD)
        pow10_list.append(p)
    
    # Compute prefix sum array
    prefix = [0] * n
    prefix[0] = a[0] % MOD
    for i in range(1, n):
        prefix[i] = (prefix[i-1] + a[i]) % MOD
    
    # Calculate sum_part1
    sum_part1 = 0
    for j in range(1, n):
        sum_part1 = (sum_part1 + prefix[j-1] * pow10_list[j]) % MOD
    
    # Calculate sum_part2
    sum_part2 = 0
    for j in range(n):
        sum_part2 = (sum_part2 + a[j] * j) % MOD
    
    total = (sum_part1 + sum_part2) % MOD
    print(total)

if __name__ == "__main__":
    main()