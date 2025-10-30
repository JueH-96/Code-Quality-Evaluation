import math

def solve():
    m = int(input())
    s1 = input()
    s2 = input()
    s3 = input()
    min_overall_time = float('inf')
    possible = False
    for digit_char in '0123456789':
        indices1 = [i for i, char in enumerate(s1) if char == digit_char]
        indices2 = [i for i, char in enumerate(s2) if char == digit_char]
        indices3 = [i for i, char in enumerate(s3) if char == digit_char]
        if not indices1 or not indices2 or not indices3:
            continue
        possible = True
        min_time_for_digit = float('inf')
        reels = [1, 2, 3]
        index_sets = {1: indices1, 2: indices2, 3: indices3}
        import itertools
        reel_permutations = list(itertools.permutations(reels))
        for p in reel_permutations:
            reel_order = list(p)
            current_indices_sets = [index_sets[reel_order[0]], index_sets[reel_order[1]], index_sets[reel_order[2]]]
            for j1 in current_indices_sets[0]:
                for j2 in current_indices_sets[1]:
                    for j3 in current_indices_sets[2]:
                        t1 = j1
                        t2_prime = j2
                        t2 = t2_prime if t2_prime > t1 else t2_prime + m
                        t3_prime = j3
                        t3 = t3_prime if t3_prime > t2 else t3_prime + m
                        current_times = {reel_order[0]: t1, reel_order[1]: t2, reel_order[2]: t3}
                        max_time = max(current_times.values())
                        min_time_for_digit = min(min_time_for_digit, max_time)
        min_overall_time = min(min_overall_time, min_time_for_digit)
        
    if not possible:
        print("-1")
    elif min_overall_time == float('inf'):
        print("-1")
    else:
        print(min_overall_time)

if __name__ == '__main__':
    solve()