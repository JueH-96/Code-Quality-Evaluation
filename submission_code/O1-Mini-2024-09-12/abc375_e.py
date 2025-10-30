# YOUR CODE HERE
import sys
import sys
from collections import defaultdict

def main():
    import sys, math
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = []
    B = []
    for i in range(N):
        A.append(int(data[1 + 2*i]))
        B.append(int(data[2 + 2*i]))
    
    sum1 = sum(B[i] for i in range(N) if A[i]==1)
    sum2 = sum(B[i] for i in range(N) if A[i]==2)
    sum3 = sum(B[i] for i in range(N) if A[i]==3)
    
    total_sum = sum1 + sum2 + sum3
    if total_sum %3 !=0:
        print(-1)
        return
    S = total_sum //3
    diff1 = sum1 - S
    diff2 = sum2 - S
    diff3 = sum3 - S
    
    # Identify excess teams
    excess_teams = []
    if diff1 >0:
        excess_teams.append((1, diff1))
    if diff2 >0:
        excess_teams.append((2, diff2))
    if diff3 >0:
        excess_teams.append((3, diff3))
    
    # Identify deficit teams
    deficit_teams = []
    if diff1 <0:
        deficit_teams.append((-1, -diff1))
    if diff2 <0:
        deficit_teams.append((-2, -diff2))
    if diff3 <0:
        deficit_teams.append((-3, -diff3))
    
    # If no excess and no deficit, already balanced
    if not excess_teams and not deficit_teams:
        print(0)
        return
    
    # Collect excess people
    excess_people = []
    for team, ex in excess_teams:
        for i in range(N):
            if A[i]==team:
                excess_people.append(B[i])
    
    # Total deficit should equal total excess
    total_excess = sum(ex for _, ex in excess_teams)
    total_deficit = sum(defi for _, defi in deficit_teams)
    if total_excess != total_deficit:
        print(-1)
        return
    
    # Now, attempt to distribute the excess to cover the deficits
    if len(deficit_teams)==1:
        D = deficit_teams[0][1]
        dp = [math.inf]*(D+1)
        dp[0]=0
        for p in excess_people:
            for j in range(D, p-1, -1):
                if dp[j-p] +1 < dp[j]:
                    dp[j] = dp[j-p] +1
        if dp[D] != math.inf:
            print(dp[D])
        else:
            print(-1)
    elif len(deficit_teams)==2:
        D1 = deficit_teams[0][1]
        D2 = deficit_teams[1][1]
        dp = {}
        dp[(0,0)] =0
        for p in excess_people:
            temp = dict(dp)
            for (j,k), cnt in dp.items():
                if j + p <= D1:
                    key = (j + p, k)
                    if key not in temp or cnt +1 < temp[key]:
                        temp[key] = cnt +1
                if k + p <= D2:
                    key = (j, k + p)
                    if key not in temp or cnt +1 < temp[key]:
                        temp[key] = cnt +1
            dp = temp
        if (D1, D2) in dp:
            print(dp[(D1,D2)])
        else:
            print(-1)
    else:
        # More than 2 deficits, not possible as there are only 3 teams
        print(-1)

if __name__ == "__main__":
    main()