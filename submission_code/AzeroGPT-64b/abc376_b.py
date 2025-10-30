from collections import defaultdict
N,Q = map(int,input().split())
initialState = [0,1]
states = defaultdict(list)
hand_pos = {"L":0, "R":1}
for i in range(Q):
    hand, position = input().split()
    position = int(position)
    states[(i,hand_pos[hand])].append(position)

def calcCost(targetPos,handPos,N):
    if targetPos == handPos:
        return 0
    else:
        direct = abs(targetPos - handPos)
        indirect = N - direct
        return min(direct, indirect)

cost = 0
current_left = 1
current_right = 2
for i in range(Q):
    for targetHand, targetPos in states[(i,0)]:
        cost += calcCost(targetPos, current_left, N)
        current_left = targetPos
    for targetHand, targetPos in states[(i,1)]:
        cost += calcCost(targetPos, current_right, N)
        current_right = targetPos
print(cost)