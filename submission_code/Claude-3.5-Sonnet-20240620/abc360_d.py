# YOUR CODE HERE
def count_passing_pairs(N, T, S, X):
    ants = [(X[i], 1 if S[i] == '1' else -1, i) for i in range(N)]
    ants.sort()
    
    stack = []
    count = 0
    
    for x, direction, index in ants:
        while stack and x - stack[-1][0] <= 2 * T:
            if direction != stack[-1][1]:
                count += 1
            stack.pop()
        
        stack.append((x, direction, index))
    
    return count

N, T = map(int, input().split())
S = input().strip()
X = list(map(int, input().split()))

print(count_passing_pairs(N, T, S, X))