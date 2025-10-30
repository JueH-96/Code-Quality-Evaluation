import sys
from collections import defaultdict

MOD = 998244353

def main():
    input = sys.stdin.read().split()
    N = int(input[0])
    K = int(input[1])
    A = list(map(int, input[2:2+N]))
    
    cnt = defaultdict(int)
    cnt[0] = 1
    prefix = 0
    sum_so_far = 1
    dp_i = 0
    
    for a in A:
        prefix += a
        invalid = cnt[prefix - K]
        dp_i = (sum_so_far - invalid) % MOD
        sum_so_far = (sum_so_far + dp_i) % MOD
        cnt[prefix] = (cnt[prefix] + dp_i) % MOD
    
    print(dp_i % MOD)

if __name__ == "__main__":
    main()