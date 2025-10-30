# YOUR CODE HERE
import sys

def find_culprit():
    input = sys.stdin.read().strip().split()
    A = int(input[0])
    B = int(input[1])
    
    # Possible suspects based on Ringo's memory
    ringo_suspects = {1, 2, 3} - {A}
    # Possible suspects based on Snuke's memory
    snuke_suspects = {1, 2, 3} - {B}
    
    # Intersection of both sets gives the possible culprit(s)
    possible_culprits = ringo_suspects & snuke_suspects
    
    if len(possible_culprits) == 1:
        print(possible_culprits.pop())
    else:
        print(-1)

find_culprit()