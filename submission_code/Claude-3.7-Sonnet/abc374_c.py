def solve():
    N = int(input())
    departments = list(map(int, input().split()))
    
    total_people = sum(departments)
    min_max_people = total_people  # Worst case: all in one group
    
    # Loop through all possible configurations (2^N)
    for config in range(1, (1 << N) - 1):  # Skip the configurations where one group is empty
        group_a_total = 0
        
        # Determine the assignment for each department
        for i in range(N):
            if (config >> i) & 1:  # Department i is assigned to Group A
                group_a_total += departments[i]
        
        group_b_total = total_people - group_a_total
        
        # Compute the maximum of both groups
        max_people = max(group_a_total, group_b_total)
        
        # Update the minimum maximum
        min_max_people = min(min_max_people, max_people)
    
    return min_max_people

# Read input and solve the problem
N = int(input())
departments = list(map(int, input().split()))
print(solve())