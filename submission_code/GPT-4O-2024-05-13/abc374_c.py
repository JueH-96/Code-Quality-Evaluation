# YOUR CODE HERE
import sys
from itertools import combinations

def min_max_lunch_break(N, K):
    total_sum = sum(K)
    min_max_people = total_sum
    
    # Generate all possible combinations of departments
    for i in range(1, N):
        for comb in combinations(K, i):
            group_a = sum(comb)
            group_b = total_sum - group_a
            max_people = max(group_a, group_b)
            min_max_people = min(min_max_people, max_people)
    
    return min_max_people

if __name__ == "__main__":
    input = sys.stdin.read().strip().split()
    N = int(input[0])
    K = list(map(int, input[1:]))
    result = min_max_lunch_break(N, K)
    print(result)