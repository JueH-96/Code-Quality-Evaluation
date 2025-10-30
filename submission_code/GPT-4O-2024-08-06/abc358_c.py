# YOUR CODE HERE
import sys
from itertools import combinations

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    stands = data[2:]
    
    # Convert each stand's string to a set of flavors it sells
    flavor_sets = []
    for stand in stands:
        flavor_set = set()
        for j in range(M):
            if stand[j] == 'o':
                flavor_set.add(j)
        flavor_sets.append(flavor_set)
    
    # We need to cover all flavors from 0 to M-1
    all_flavors = set(range(M))
    
    # Try combinations of stands from size 1 to N
    for r in range(1, N + 1):
        for combo in combinations(range(N), r):
            # Union of all flavors sold by the selected stands
            combined_flavors = set()
            for idx in combo:
                combined_flavors.update(flavor_sets[idx])
            
            # Check if this combination covers all flavors
            if combined_flavors == all_flavors:
                print(r)
                return