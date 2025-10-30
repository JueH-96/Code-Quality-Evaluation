# YOUR CODE HERE
import sys

def can_complete_moves(N, M, H, K, S, items):
    x, y = 0, 0
    items_set = set(items)
    
    for i in range(N):
        if S[i] == 'R':
            x += 1
        elif S[i] == 'L':
            x -= 1
        elif S[i] == 'U':
            y += 1
        elif S[i] == 'D':
            y -= 1
        
        H -= 1
        if H < 0:
            return "No"
        
        if (x, y) in items_set and H < K:
            H = K
            items_set.remove((x, y))
    
    return "Yes"

input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
H = int(data[2])
K = int(data[3])
S = data[4]
items = [(int(data[5 + 2 * i]), int(data[6 + 2 * i])) for i in range(M)]

print(can_complete_moves(N, M, H, K, S, items))