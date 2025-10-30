import sys
from collections import defaultdict

def main():
    N = int(sys.stdin.readline())
    people = []
    S1 = 0
    S2 = 0
    S3 = 0
    for _ in range(N):
        A, B = map(int, sys.stdin.readline().split())
        people.append((A, B))
        if A == 1:
            S1 += B
        elif A == 2:
            S2 += B
        else:
            S3 += B
    total_B = S1 + S2 + S3
    if total_B % 3 != 0:
        print(-1)
        return
    T = total_B // 3
    initial_d1 = T - S1
    initial_d2 = T - S2
    # Check if initial deltas are within valid range
    if not (-500 <= initial_d1 <= 500 and -500 <= initial_d2 <= 500):
        print(-1)
        return
    # Initialize DP as a dictionary
    dp = defaultdict(lambda: float('inf'))
    dp[(initial_d1, initial_d2)] = 0
    for (X, B) in people:
        next_dp = defaultdict(lambda: float('inf'))
        for (current_d1, current_d2), current_moves in dp.items():
            for Y in [1, 2, 3]:
                if Y == X:
                    new_d1 = current_d1
                    new_d2 = current_d2
                    new_moves = current_moves
                else:
                    if X == 1:
                        if Y == 2:
                            new_d1 = current_d1 + B
                            new_d2 = current_d2 - B
                        elif Y == 3:
                            new_d1 = current_d1 + B
                            new_d2 = current_d2
                    elif X == 2:
                        if Y == 1:
                            new_d1 = current_d1 - B
                            new_d2 = current_d2 + B
                        elif Y == 3:
                            new_d1 = current_d1
                            new_d2 = current_d2 + B
                    elif X == 3:
                        if Y == 1:
                            new_d1 = current_d1 - B
                            new_d2 = current_d2
                        elif Y == 2:
                            new_d1 = current_d1
                            new_d2 = current_d2 - B
                    new_moves = current_moves + 1
                # Check if new_d1 and new_d2 are within valid range
                if (-500 <= new_d1 <= 500) and (-500 <= new_d2 <= 500):
                    key = (new_d1, new_d2)
                    if new_moves < next_dp[key]:
                        next_dp[key] = new_moves
        dp = next_dp
    # Check the target state (0,0)
    if (0, 0) in dp:
        print(dp[(0, 0)])
    else:
        print(-1)

if __name__ == "__main__":
    main()