# YOUR CODE HERE
N, X = map(int, input().split())
S_list = list(map(int, input().split()))
total = sum(S for S in S_list if S <= X)
print(total)