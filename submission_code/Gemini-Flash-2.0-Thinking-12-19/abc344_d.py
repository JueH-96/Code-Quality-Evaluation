import math

def solve():
    target_string = input()
    n_bags = int(input())
    bags = []
    for _ in range(n_bags):
        line = input().split()
        bag_size = int(line[0])
        strings_in_bag = line[1:]
        bags.append(strings_in_bag)
    
    target_len = len(target_string)
    dp = {}
    dp[0, 0] = 0
    
    for i in range(1, n_bags + 1):
        for j in range(target_len + 1):
            dp[i, j] = math.inf
            # Option 1: Do nothing
            if (i - 1, j) in dp:
                dp[i, j] = min(dp[i, j], dp[i - 1, j])
            
            # Option 2: Select a string from bag i
            for s in bags[i-1]:
                s_len = len(s)
                if j >= s_len:
                    prefix_len = j - s_len
                    if (i - 1, prefix_len) in dp and dp[i-1, prefix_len] != math.inf:
                        if target_string[prefix_len:j] == s:
                            dp[i, j] = min(dp[i, j], dp[i - 1, prefix_len] + 1)
                            
    result = dp.get((n_bags, target_len), math.inf)
    if result == math.inf:
        print("-1")
    else:
        print(result)

if __name__ == '__main__':
    solve()