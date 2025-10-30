def solve():
    target_string = input()
    n_bags = int(input())
    bags = []
    for _ in range(n_bags):
        line = input().split()
        count = int(line[0])
        strings = line[1:]
        bags.append(strings)
    
    target_len = len(target_string)
    dp = {}
    dp[0] = 0
    
    for i in range(n_bags):
        next_dp = {}
        for current_prefix_len in dp:
            current_cost = dp[current_prefix_len]
            if current_prefix_len not in next_dp or current_cost < next_dp[current_prefix_len]:
                next_dp[current_prefix_len] = current_cost
                
            for s in bags[i]:
                new_prefix_len = calculate_new_prefix_length(current_prefix_len, s, target_string)
                cost_with_string = current_cost + 1
                if new_prefix_len not in next_dp or cost_with_string < next_dp[new_prefix_len]:
                    next_dp[new_prefix_len] = cost_with_string
                    
        dp = next_dp
        
    if target_len in dp:
        print(dp[target_len])
    else:
        print(-1)

def calculate_new_prefix_length(current_prefix_length, appended_string, target_string):
    target_len = len(target_string)
    appended_len = len(appended_string)
    match_len = 0
    while current_prefix_length + match_len < target_len and match_len < appended_len:
        if target_string[current_prefix_length + match_len] == appended_string[match_len]:
            match_len += 1
        else:
            break
    return current_prefix_length + match_len

if __name__ == '__main__':
    solve()