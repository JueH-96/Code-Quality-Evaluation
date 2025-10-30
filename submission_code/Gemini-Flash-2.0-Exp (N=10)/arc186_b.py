def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    mod = 998244353
    
    adj = [[] for _ in range(n)]
    for i in range(1, n):
        if a[i] > 0:
            adj[a[i]].append(i)
    
    
    def count_permutations(current_perm, remaining_nums, current_index):
        if current_index == n:
            return 1
        
        count = 0
        for num in remaining_nums:
            valid = True
            
            # Condition 1: P_j > P_i for any integer j with A_i < j < i
            for j in range(a[current_index], current_index):
              if j > 0 and j < len(current_perm) and current_perm[j] != 0 and current_perm[j] < num:
                valid = False
                break
            
            if not valid:
              continue
            
            # Condition 2: P_{A_i} < P_i if A_i > 0
            if a[current_index] > 0 and current_perm[a[current_index]] != 0 and current_perm[a[current_index]] > num:
                valid = False
                
            if not valid:
              continue
            
            
            
            next_perm = current_perm[:]
            next_perm.append(num)
            next_remaining = remaining_nums[:]
            next_remaining.remove(num)
            
            count = (count + count_permutations(next_perm, next_remaining, current_index + 1)) % mod
        
        return count

    
    
    initial_perm = []
    initial_remaining = list(range(1, n + 1))
    
    result = count_permutations(initial_perm, initial_remaining, 0)
    print(result)

solve()