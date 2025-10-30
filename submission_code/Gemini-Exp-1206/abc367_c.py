def solve():
    n, k = map(int, input().split())
    r = list(map(int, input().split()))
    
    ans = []
    
    def backtrack(index, current_sequence, current_sum):
        if index == n:
            if current_sum % k == 0:
                ans.append(current_sequence.copy())
            return
        
        for i in range(1, r[index] + 1):
            current_sequence.append(i)
            backtrack(index + 1, current_sequence, current_sum + i)
            current_sequence.pop()
    
    backtrack(0, [], 0)
    
    for seq in ans:
        print(*seq)

solve()