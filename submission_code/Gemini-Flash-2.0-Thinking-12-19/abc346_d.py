def solve():
    n = int(input())
    s = input()
    costs = list(map(int, input().split()))
    min_total_cost = float('inf')
    
    for i in range(n - 1):
        for identical_value in ['0', '1']:
            target_t = [''] * n
            # Construct target string T^{(i, identical_value)}
            for j in range(i + 1):
                if identical_value == '0':
                    target_t[j] = '1' if (i - j) % 2 == 1 else '0'
                else:
                    target_t[j] = '0' if (i - j) % 2 == 1 else '1'
            target_t[i+1] = identical_value
            for j in range(i + 2, n):
                if identical_value == '0':
                    target_t[j] = '1' if (j - (i + 1)) % 2 == 1 else '0'
                else:
                    target_t[j] = '0' if (j - (i + 1)) % 2 == 1 else '1'
            
            current_cost = 0
            for j in range(n):
                if s[j] != target_t[j]:
                    current_cost += costs[j]
            
            # Check if it's a good string
            is_good = True
            identical_pairs_count = 0
            for j in range(n - 1):
                if target_t[j] == target_t[j+1]:
                    identical_pairs_count += 1
            if identical_pairs_count != 1:
                is_good = False
            
            if is_good:
                min_total_cost = min(min_total_cost, current_cost)
                
    print(min_total_cost)

if __name__ == '__main__':
    solve()