# YOUR CODE HERE
def run_solution():
    N, M = map(int, input().split())

    # has_taro_in_family[family_idx] is True if family_idx already has a Taro.
    # Families are 1-indexed, so list size is N+1. Index 0 is unused.
    has_taro_in_family = [False] * (N + 1)

    for _ in range(M):
        # Read baby's family and gender
        line_parts = input().split()
        family_id = int(line_parts[0])
        gender = line_parts[1]  # 'M' or 'F'
        
        # Determine if this baby is named Taro
        baby_is_taro = False  # Assume not Taro by default
        
        if gender == 'M':  # Only males can be Taro
            # Check if this family has already had its eldest son (Taro)
            if not has_taro_in_family[family_id]:
                # This male baby is the first in this family
                baby_is_taro = True
                # Mark that this family now has its Taro
                has_taro_in_family[family_id] = True
        
        # Output result for this baby
        if baby_is_taro:
            print("Yes")
        else:
            print("No")

run_solution()