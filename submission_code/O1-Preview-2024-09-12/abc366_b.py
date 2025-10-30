# YOUR CODE HERE
N = int(input())
S_list = [input() for _ in range(N)]
M = max(len(s) for s in S_list)
T_j_list = [[] for _ in range(M)]
for i in range(N):
    s = S_list[i]
    r = N - i - 1
    for j in range(len(s)):
        while len(T_j_list[j]) <= r:
            T_j_list[j].append('*')
        T_j_list[j][r] = s[j]
for T_j in T_j_list:
    while T_j and T_j[-1] == '*':
        T_j.pop()
for T_j in T_j_list:
    print(''.join(T_j))