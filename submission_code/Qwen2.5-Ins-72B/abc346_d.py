# YOUR CODE HERE
import sys

def min_cost_to_good_string(N, S, C):
    def cost_to_good_string(S, C, flip_first):
        cost = 0
        if flip_first:
            S = '1' if S[0] == '0' else '0' + S[1:]
            cost += C[0]
        
        for i in range(1, N):
            if S[i] == S[i-1]:
                if i < N-1 and S[i] == S[i+1]:
                    if C[i] < C[i-1]:
                        cost += C[i]
                        S = S[:i] + ('1' if S[i] == '0' else '0') + S[i+1:]
                    else:
                        cost += C[i-1]
                        S = S[:i-1] + ('1' if S[i-1] == '0' else '0') + S[i:]
                else:
                    cost += C[i-1]
                    S = S[:i-1] + ('1' if S[i-1] == '0' else '0') + S[i:]
        
        return cost
    
    min_cost = float('inf')
    for flip_first in [True, False]:
        min_cost = min(min_cost, cost_to_good_string(S, C, flip_first))
    
    return min_cost

input = sys.stdin.read
data = input().split()

N = int(data[0])
S = data[1]
C = list(map(int, data[2:]))

print(min_cost_to_good_string(N, S, C))