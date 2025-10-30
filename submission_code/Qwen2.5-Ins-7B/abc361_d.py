# YOUR CODE HERE
from collections import deque

def solve():
    N = int(input())
    S = input()
    T = input()
    dp = [-1] * (1 << N)
    dp[0] = 0
    queue = deque([0])
    
    while queue:
        state = queue.popleft()
        for i in range(N):
            if state & (1 << i):
                next_state = state ^ (1 << i)
                for j in range(i + 1, N):
                    if state & (1 << j):
                        next_state2 = next_state ^ (1 << j)
                        if next_state2 == 0 or (next_state2 >> (N - 1)) & 1 == 1:
                            if dp[next_state2] == -1 or dp[next_state2] > dp[state] + 1:
                                dp[next_state2] = dp[state] + 1
                                queue.append(next_state2)
    
    ans = dp[(1 << N) - 1]
    if ans == -1:
        print(-1)
    else:
        print(ans)

solve()