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
    
    # Calculate current team strengths
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
    # We will try all possible combinations of switches
    # Since N is small (up to 100), but the number of possible switches is large, we need a smarter approach
    
    # Let's consider that for each person, we can choose to switch to any of the other two teams
    # We need to find a combination of switches that makes the sums equal to target
    
    # To manage this, we can represent the problem as a state where we have the current sums of the teams
    # and we try to find the minimal number of switches to reach the target sums
    
    # We can use a BFS approach where each state is represented by the current sums of the teams and the number of switches
    
    from collections import deque
    
    initial_state = (sum1, sum2, sum3, 0)
    visited = set()
    queue = deque()
    queue.append(initial_state)
    visited.add((sum1, sum2, sum3))
    
    while queue:
        s1, s2, s3, switches = queue.popleft()
        if s1 == target and s2 == target and s3 == target:
            print(switches)
            return
        for i in range(N):
            current_team = A[i]
            if current_team == 1:
                new_s1 = s1 - B[i]
                new_s2 = s2 + B[i]
                new_s3 = s3
                new_state = (new_s1, new_s2, new_s3)
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_s1, new_s2, new_s3, switches + 1))
                new_s1 = s1 - B[i]
                new_s2 = s2
                new_s3 = s3 + B[i]
                new_state = (new_s1, new_s2, new_s3)
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_s1, new_s2, new_s3, switches + 1))
            elif current_team == 2:
                new_s1 = s1 + B[i]
                new_s2 = s2 - B[i]
                new_s3 = s3
                new_state = (new_s1, new_s2, new_s3)
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_s1, new_s2, new_s3, switches + 1))
                new_s1 = s1
                new_s2 = s2 - B[i]
                new_s3 = s3 + B[i]
                new_state = (new_s1, new_s2, new_s3)
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_s1, new_s2, new_s3, switches + 1))
            elif current_team == 3:
                new_s1 = s1 + B[i]
                new_s2 = s2
                new_s3 = s3 - B[i]
                new_state = (new_s1, new_s2, new_s3)
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_s1, new_s2, new_s3, switches + 1))
                new_s1 = s1
                new_s2 = s2 + B[i]
                new_s3 = s3 - B[i]
                new_state = (new_s1, new_s2, new_s3)
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_s1, new_s2, new_s3, switches + 1))
    
    print(-1)

if __name__ == "__main__":
    main()