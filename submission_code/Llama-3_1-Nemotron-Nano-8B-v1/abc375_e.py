import sys
from collections import defaultdict

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx +=1
    people = []
    sum_total = 0
    sum1 = 0
    sum2 = 0
    sum3 = 0
    for _ in range(N):
        A = int(input[idx])-1  # 0-based for easier handling
        B = int(input[idx+1])
        people.append((A, B))
        sum_total += B
        if A == 0:
            sum1 += B
        elif A == 1:
            sum2 += B
        else:
            sum3 += B
        idx +=2
    
    if sum_total % 3 != 0:
        print(-1)
        return
    T = sum_total // 3
    
    # DP state is (sum1, sum2), sum3 = sum_total - sum1 - sum2
    dp = defaultdict(lambda: float('inf'))
    initial_sum1 = sum1
    initial_sum2 = sum2
    dp[(initial_sum1, initial_sum2)] = 0
    
    for a, b in people:
        new_dp = defaultdict(lambda: float('inf'))
        for (s1, s2), moves in dp.items():
            for team in [0, 1, 2]:  # 0: team1, 1: team2, 2: team3
                new_s1 = s1
                new_s2 = s2
                if team == 0:
                    new_s1 += b
                elif team == 1:
                    new_s2 += b
                # else, team ==2, no change to s1, s2
                new_moves = moves + (0 if team == a else 1)
                new_s1_val = new_s1
                new_s2_val = new_s2
                if new_s1_val < 0 or new_s2_val < 0 or (sum_total - new_s1_val - new_s2_val) < 0:
                    continue
                key = (new_s1_val, new_s2_val)
                if new_moves < new_dp[key]:
                    new_dp[key] = new_moves
        dp = new_dp
        if not dp:
            print(-1)
            return
    
    target = T
    min_moves = float('inf')
    for (s1, s2), moves in dp.items():
        if s1 == target and s2 == target and (sum_total - s1 - s2) == target:
            if moves < min_moves:
                min_moves = moves
    if min_moves != float('inf'):
        print(min_moves)
    else:
        print(-1)

if __name__ == '__main__':
    main()