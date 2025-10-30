# YOUR CODE HERE
N, M = map(int, input().split())
A = list(map(int, input().split()))

S = []
for _ in range(N):
    S.append(input())

# Compute s_i (current total scores) and unsolved problems for each player
s_i_list = []
unsolved_problems_list = []
for i in range(N):
    solved_score = 0
    unsolved_problems = []
    for idx, ch in enumerate(S[i]):
        if ch == 'o':
            solved_score += A[idx]
        else:
            unsolved_problems.append(A[idx])
    s_i = solved_score + (i + 1)  # player i's bonus is (i + 1)
    s_i_list.append(s_i)
    unsolved_problems.sort(reverse=True)
    unsolved_problems_list.append(unsolved_problems)

# For each player, compute the minimum number of unsolved problems to solve
for i in range(N):
    s_i = s_i_list[i]
    unsolved_problems = unsolved_problems_list[i]
    s_j_max = max(s_j_list for idx, s_j_list in enumerate(s_i_list) if idx != i)
    s_i_new = s_i
    k = 0
    while True:
        if s_i_new > s_j_max:
            print(k)
            break
        if k == len(unsolved_problems):
            # According to the problem statement, answer always exists
            print(k)
            break
        s_i_new += unsolved_problems[k]
        k += 1