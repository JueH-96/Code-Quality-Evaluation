def solve():
    N = int(input())
    A = list(map(int, input().split()))
    
    initial = tuple(i % 2 for i in range(1, N+1))
    target = tuple(A)
    
    memo = {}
    
    def count_ways(state):
        if state == target:
            return 1
        
        if state in memo:
            return memo[state]
        
        total = 0
        
        for l in range(N):
            for r in range(l+2, N):
                if state[l] == state[r]:
                    # Check if all positions between l and r have different values
                    valid = True
                    for i in range(l+1, r):
                        if state[i] == state[l]:
                            valid = False
                            break
                    
                    if valid:
                        # Perform the operation
                        new_state = list(state)
                        for i in range(l+1, r):
                            new_state[i] = state[l]
                        new_state = tuple(new_state)
                        
                        total = (total + count_ways(new_state)) % 998244353
        
        memo[state] = total
        return total
    
    return count_ways(initial)

print(solve())