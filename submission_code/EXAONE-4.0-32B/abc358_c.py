import sys

def main():
    data = sys.stdin.read().splitlines()
    if not data:
        print(0)
        return
        
    n, m = map(int, data[0].split())
    strings = []
    for i in range(1, 1 + n):
        strings.append(data[i].strip())
    
    masks = []
    for s in strings:
        mask_val = 0
        for j, char in enumerate(s):
            if char == 'o':
                mask_val |= (1 << j)
        masks.append(mask_val)
        
    total_states = 1 << m
    INF = 100
    dp = [INF] * total_states
    dp[0] = 0
    
    for mask in masks:
        new_dp = dp.copy()
        for state in range(total_states):
            if dp[state] == INF:
                continue
            new_state = state | mask
            if dp[state] + 1 < new_dp[new_state]:
                new_dp[new_state] = dp[state] + 1
        dp = new_dp
        
    full_state = (1 << m) - 1
    print(dp[full_state])

if __name__ == "__main__":
    main()