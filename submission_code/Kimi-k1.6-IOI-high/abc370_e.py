import sys
from collections import defaultdict

MOD = 998244353

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr +=1
    K = int(input[ptr])
    ptr +=1
    A = list(map(int, input[ptr:ptr+N]))
    
    sum_map = defaultdict(int)
    sum_map[0] = 1
    current_sum = 0
    total = 1
    ans = 0
    
    for num in A:
        current_sum += num
        forbidden = current_sum - K
        bad = sum_map.get(forbidden, 0)
        dp_i = (total - bad) % MOD
        sum_map[current_sum] = (sum_map[current_sum] + dp_i) % MOD
        total = (total + dp_i) % MOD
        ans = dp_i
    
    print(ans % MOD)

if __name__ == "__main__":
    main()