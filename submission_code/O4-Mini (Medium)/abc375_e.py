import sys
import threading

def main():
    import sys

    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    A = [0] * N
    B = [0] * N
    sum_total = 0
    for i in range(N):
        ai = int(next(it))
        bi = int(next(it))
        A[i] = ai - 1  # zero-based teams 0,1,2
        B[i] = bi
        sum_total += bi

    # If total strength not divisible by 3, impossible
    if sum_total % 3 != 0:
        print(-1)
        return
    target = sum_total // 3

    # dp[x][y] = max number of people kept (not moved) when
    # team0 sum = x, team1 sum = y, and team2 sum <= target
    # We maintain sum2 <= target with a check on team2 assignment.
    # Initialize dp
    # -1 means unreachable
    dp = [[-1] * (target + 1) for _ in range(target + 1)]
    dp[0][0] = 0
    sumB = 0  # sum of B processed so far

    for i in range(N):
        bi = B[i]
        ai = A[i]
        sumB_prev = sumB
        sumB += bi
        dp2 = [[-1] * (target + 1) for _ in range(target + 1)]
        # transition for each reachable state
        for x in range(target + 1):
            row_dp = dp[x]
            for y in range(target + 1):
                val = row_dp[y]
                if val < 0:
                    continue
                # sum for team2 so far
                sum2_pre = sumB_prev - x - y
                # assign to team0
                nx = x + bi
                if nx <= target:
                    kept = val + (1 if ai == 0 else 0)
                    if kept > dp2[nx][y]:
                        dp2[nx][y] = kept
                # assign to team1
                ny = y + bi
                if ny <= target:
                    kept = val + (1 if ai == 1 else 0)
                    if kept > dp2[x][ny]:
                        dp2[x][ny] = kept
                # assign to team2, only if sum2_pre + bi <= target
                if sum2_pre + bi <= target:
                    kept = val + (1 if ai == 2 else 0)
                    if kept > dp2[x][y]:
                        dp2[x][y] = kept
        dp = dp2

    # Check final dp[target][target]
    best_keep = dp[target][target]
    if best_keep < 0:
        print(-1)
    else:
        # minimum moves = total people - people who stayed
        print(N - best_keep)

if __name__ == "__main__":
    main()