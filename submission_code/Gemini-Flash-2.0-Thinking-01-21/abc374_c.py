def solve():
    n = int(input())
    k_values = list(map(int, input().split()))
    total_sum = sum(k_values)
    target_sum = total_sum // 2
    possible_group_a_sums = {0}
    for department_size in k_values:
        next_sums = set()
        for current_sum in possible_group_a_sums:
            next_sums.add(current_sum)
            next_sums.add(current_sum + department_size)
        possible_group_a_sums = next_sums
    
    max_group_a_sum_achieved = 0
    for s in possible_group_a_sums:
        if s <= target_sum:
            max_group_a_sum_achieved = max(max_group_a_sum_achieved, s)
        elif s > target_sum:
            max_group_a_sum_achieved = max(max_group_a_sum_achieved, s)
            
    min_max_people = float('inf')
    for group_a_sum in possible_group_a_sums:
        group_b_sum = total_sum - group_a_sum
        current_max_people = max(group_a_sum, group_b_sum)
        min_max_people = min(min_max_people, current_max_people)
        
    print(min_max_people)

if __name__ == '__main__':
    solve()