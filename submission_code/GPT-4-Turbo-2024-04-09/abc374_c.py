import sys
input = sys.stdin.read

def solve():
    data = input().split()
    N = int(data[0])
    K = list(map(int, data[1:]))
    
    # Since N is at most 20, we can consider all 2^N possible ways to divide the departments
    # into two groups (by considering each subset of departments as one group).
    
    min_max_people = float('inf')
    
    # There are 2^N possible subsets of departments
    total_subsets = 1 << N
    
    for subset in range(total_subsets):
        sum_group_A = 0
        sum_group_B = 0
        
        for i in range(N):
            if subset & (1 << i):
                sum_group_A += K[i]
            else:
                sum_group_B += K[i]
        
        # We want to minimize the maximum number of people in either group
        max_people = max(sum_group_A, sum_group_B)
        min_max_people = min(min_max_people, max_people)
    
    print(min_max_people)