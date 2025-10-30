# YOUR CODE HERE
import sys
from heapq import heappush, heappop

input = sys.stdin.readline

class Node:
    def __init__(self, cost, prob, exp):
        self.cost = cost
        self.prob = prob
        self.exp = exp
    def __lt__(self, other):
        return self.exp < other.exp

def solve():
    N, M = map(int, input().split())
    wheels = []
    for _ in range(N):
        C, P = map(int, input().split())
        wheel = [C] + list(map(int, input().split()))
        wheels.append(wheel)
    
    pq = []
    for i in range(N):
        heappush(pq, Node(wheels[i][0], 1, wheels[i][1]))
    
    while pq:
        node = heappop(pq)
        if node.exp >= M:
            break
        for i in range(1, wheels[node.cost-1][1]+1):
            new_node = Node(wheels[node.cost-1][0], node.prob * (1/wheels[node.cost-1][1]), node.exp + wheels[node.cost-1][i])
            heappush(pq, new_node)
    
    print(node.exp)

solve()