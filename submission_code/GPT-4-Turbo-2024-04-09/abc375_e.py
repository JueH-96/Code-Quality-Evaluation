import sys
input = sys.stdin.read

def solve():
    data = input().split()
    N = int(data[0])
    A = []
    B = []
    
    index = 1
    for _ in range(N):
        A.append(int(data[index]))
        B.append(int(data[index + 1]))
        index += 2
    
    # Calculate the initial strength of each team
    strength = [0, 0, 0]  # strength[0] for team 1, strength[1] for team 2, strength[2] for team 3
    for i in range(N):
        team = A[i] - 1
        strength[team] += B[i]
    
    total_strength = sum(strength)
    
    # Check if the total strength can be evenly divided by 3
    if total_strength % 3 != 0:
        print(-1)
        return
    
    target_strength = total_strength // 3
    
    # If all teams already have the target strength
    if strength[0] == strength[1] == strength[2] == target_strength:
        print(0)
        return
    
    # Calculate how much each team needs to gain or lose
    need = [target_strength - s for s in strength]
    
    # If all needs sum to zero and each need is feasible
    if sum(need) != 0:
        print(-1)
        return
    
    # We need to balance the teams by moving strengths
    # We will consider the excess and deficit in each team
    excess = [max(0, -need[i]) for i in range(3)]
    deficit = [max(0, need[i]) for i in range(3)]
    
    # We need to find the minimum number of moves to balance the excess and deficit
    # We can think of this as a flow problem where we want to minimize the number of people moved
    
    # We will try to greedily match excess from one team to deficit in another
    moves = 0
    for i in range(3):
        for j in range(3):
            if i != j:
                # Amount we can transfer from team i to team j
                transfer = min(excess[i], deficit[j])
                excess[i] -= transfer
                deficit[j] -= transfer
                moves += transfer
    
    # If there's still any deficit left, it means it's impossible
    if any(deficit):
        print(-1)
    else:
        print(moves)