def solve():
    n = int(input())
    s_list = []
    for _ in range(n):
        s_list.append(input())
    
    results = []
    for k in range(1, n + 1):
        current_s = s_list[k-1]
        target_strings = [""] # empty string is always a target
        for j in range(1, k):
            target_strings.append(s_list[j-1])
        min_cost = len(current_s) # cost to empty string
        for target_s in target_strings[1:]: # iterate through S_1, ..., S_{k-1}
            lcp_len = 0
            for i in range(min(len(current_s), len(target_s))):
                if current_s[i] == target_s[i]:
                    lcp_len += 1
                else:
                    break
            cost = len(current_s) + len(target_s) - 2 * lcp_len
            min_cost = min(min_cost, cost)
        results.append(min_cost)
        
    for result in results:
        print(result)

if __name__ == '__main__':
    solve()