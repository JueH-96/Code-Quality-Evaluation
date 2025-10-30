def is_possible(N, M, S, X):
    available_plain = M
    available_logo = X
    worn_plain = 0
    worn_logo = 0
    
    for day in S:
        if day == '0':
            # Wash all worn T-shirts
            available_plain += worn_plain
            worn_plain = 0
            available_logo += worn_logo
            worn_logo = 0
        elif day == '1':
            # Try to wear plain T-shirt first
            if available_plain > 0:
                available_plain -= 1
                worn_plain += 1
            elif available_logo > 0:
                available_logo -= 1
                worn_logo += 1
            else:
                # Cannot wear any T-shirt for '1'
                return False
        elif day == '2':
            # Must wear logo T-shirt
            if available_logo > 0:
                available_logo -= 1
                worn_logo += 1
            else:
                # Cannot wear logo T-shirt for '2'
                return False
    return True

def min_logo_tshirts(N, M, S):
    left = 0
    right = N
    result = 0
    while left <= right:
        mid = (left + right) // 2
        if is_possible(N, M, S, mid):
            result = mid
            right = mid - 1
        else:
            left = mid + 1
    return result

# Read input
N, M = map(int, input().split())
S = input().strip()

# Compute and print the result
print(min_logo_tshirts(N, M, S))