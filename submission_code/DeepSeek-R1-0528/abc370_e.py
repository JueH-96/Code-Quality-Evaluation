mod = 998244353
import sys
from collections import defaultdict

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:2+n]))
    
    total_dp = 1
    prefix_dict = defaultdict(int)
    prefix_dict[0] = 1
    current_prefix = 0
    
    for i in range(n):
        current_prefix += A[i]
        to_sub = prefix_dict[current_prefix - K]
        dp_next = (total_dp - to_sub) % mod
        total_dp = (total_dp + dp_next) % mod
        prefix_dict[current_prefix] = (prefix_dict[current_prefix] + dp_next) % mod
        
    print(dp_next % mod)

if __name__ == "__main__":
    main()