mod = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:2+N]))
    
    prefix_map = {}
    prefix_map[0] = 1
    pre_sum = 0
    total = 1  # dp[0]
    result = 0
    
    for a in A:
        pre_sum += a
        required = pre_sum - K
        subtract = prefix_map.get(required, 0)
        current_dp = (total - subtract) % mod
        
        # Update prefix_map with the current pre_sum and current_dp
        if pre_sum in prefix_map:
            prefix_map[pre_sum] = (prefix_map[pre_sum] + current_dp) % mod
        else:
            prefix_map[pre_sum] = current_dp
        
        # Update total for the next iteration
        total = (total + current_dp) % mod
    
    print(current_dp % mod)

if __name__ == "__main__":
    main()