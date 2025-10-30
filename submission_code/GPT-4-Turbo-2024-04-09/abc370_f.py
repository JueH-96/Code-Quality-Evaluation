def can_divide_into_k_groups_with_min_sum(pieces, k, min_sum):
    current_sum = 0
    groups_formed = 0
    for piece in pieces:
        if current_sum + piece >= min_sum:
            groups_formed += 1
            current_sum = 0
        else:
            current_sum += piece
    return groups_formed >= k

def find_max_min_sum(pieces, k):
    left, right = 0, sum(pieces)
    while left < right:
        mid = (left + right + 1) // 2
        if can_divide_into_k_groups_with_min_sum(pieces, k, mid):
            left = mid
        else:
            right = mid - 1
    return left

def count_undivided_cut_lines(pieces, k, max_min_sum):
    current_sum = 0
    groups_formed = 0
    undivided_count = 0
    n = len(pieces)
    
    for i in range(n):
        if current_sum + pieces[i] >= max_min_sum:
            if current_sum + pieces[i] == max_min_sum:
                undivided_count += 1
            groups_formed += 1
            current_sum = 0
        else:
            current_sum += pieces[i]
    
    # Since the last group wraps around to the first piece, we need to check the last segment
    if current_sum > 0:
        undivided_count -= 1  # The last segment between last and first piece is divided
    
    return undivided_count

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
K = int(data[1])
A = list(map(int, data[2:]))

max_min_sum = find_max_min_sum(A, K)
undivided_cut_lines = count_undivided_cut_lines(A, K, max_min_sum)

print(max_min_sum, undivided_cut_lines)