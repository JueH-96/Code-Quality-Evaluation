# YOUR CODE HERE
def solve():
    N, X = map(int, input().split())
    foods = []
    for _ in range(N):
        v, a, c = map(int, input().split())
        foods.append((v-1, a, c))  # Convert to 0-indexed
    
    def can_achieve(target):
        # Check if we can achieve at least 'target' units of each vitamin
        # Try all possible subsets of foods
        for mask in range(1 << N):
            vitamins = [0, 0, 0]
            calories = 0
            
            for i in range(N):
                if mask & (1 << i):
                    v, a, c = foods[i]
                    vitamins[v] += a
                    calories += c
            
            if calories <= X and all(vitamins[i] >= target for i in range(3)):
                return True
        
        return False
    
    # Binary search on the answer
    left, right = 0, 200000
    answer = 0
    
    while left <= right:
        mid = (left + right) // 2
        if can_achieve(mid):
            answer = mid
            left = mid + 1
        else:
            right = mid - 1
    
    print(answer)

solve()