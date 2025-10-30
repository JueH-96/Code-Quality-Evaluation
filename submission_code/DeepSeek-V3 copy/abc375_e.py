# YOUR CODE HERE
import sys
from itertools import combinations

def main():
    N = int(sys.stdin.readline())
    A = []
    B = []
    for _ in range(N):
        a, b = map(int, sys.stdin.readline().split())
        A.append(a)
        B.append(b)
    
    # Calculate total strength
    total = sum(B)
    if total % 3 != 0:
        print(-1)
        return
    
    target = total // 3
    
    # Initialize team strengths
    team1 = []
    team2 = []
    team3 = []
    for i in range(N):
        if A[i] == 1:
            team1.append(B[i])
        elif A[i] == 2:
            team2.append(B[i])
        else:
            team3.append(B[i])
    
    sum1 = sum(team1)
    sum2 = sum(team2)
    sum3 = sum(team3)
    
    if sum1 == target and sum2 == target and sum3 == target:
        print(0)
        return
    
    # Now, we need to find the minimal number of switches
    # We will try all possible ways to reassign people to teams
    # Since N is small (up to 100), but 3^100 is too large, we need a smarter approach
    
    # Let's consider that for each person, we can choose to keep them in their current team or switch to one of the other two teams
    # We need to find a combination of switches that makes the sums equal to target
    
    # To manage this, we can use dynamic programming
    # We will represent the state as (sum1, sum2, sum3, number of switches)
    # We will try to reach the target sums with the minimal number of switches
    
    # Initialize DP
    # We will use a dictionary to store the minimal number of switches for each possible (sum1, sum2, sum3)
    dp = {}
    dp[(sum1, sum2, sum3)] = 0
    
    # We will process each person one by one
    for i in range(N):
        new_dp = {}
        for state, switches in dp.items():
            s1, s2, s3 = state
            # Option 1: keep the current team
            new_s1 = s1
            new_s2 = s2
            new_s3 = s3
            if A[i] == 1:
                new_s1 -= B[i]
            elif A[i] == 2:
                new_s2 -= B[i]
            else:
                new_s3 -= B[i]
            # Now, assign to team 1, 2, or 3
            # Assign to team 1
            new_state1 = (new_s1 + B[i], new_s2, new_s3)
            if new_state1 not in new_dp or new_dp[new_state1] > switches + (A[i] != 1):
                new_dp[new_state1] = switches + (A[i] != 1)
            # Assign to team 2
            new_state2 = (new_s1, new_s2 + B[i], new_s3)
            if new_state2 not in new_dp or new_dp[new_state2] > switches + (A[i] != 2):
                new_dp[new_state2] = switches + (A[i] != 2)
            # Assign to team 3
            new_state3 = (new_s1, new_s2, new_s3 + B[i])
            if new_state3 not in new_dp or new_dp[new_state3] > switches + (A[i] != 3):
                new_dp[new_state3] = switches + (A[i] != 3)
        dp = new_dp
    
    # Now, check if the target state is in dp
    target_state = (target, target, target)
    if target_state in dp:
        print(dp[target_state])
    else:
        print(-1)

if __name__ == "__main__":
    main()