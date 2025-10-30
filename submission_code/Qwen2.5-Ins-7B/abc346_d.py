# N = int(input())
# S = input()
# C = list(map(int, input().split()))

N = 5
S = "00011"
C = [3, 9, 2, 6, 4]

def min_cost_to_good_string(N, S, C):
    min_cost = float('inf')
    for i in range(1, N):
        if S[i] == S[i-1]:
            continue
        cost = 0
        for j in range(N):
            if j == i-1 or j == i:
                cost += C[j]
            else:
                cost += min(C[j], C[j-1])
        min_cost = min(min_cost, cost)
    return min_cost

print(min_cost_to_good_string(N, S, C))