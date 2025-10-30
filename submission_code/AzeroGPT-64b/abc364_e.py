from itertools import accumulate
from bisect import bisect_left

def solve(N, X, Y, dishes):
    # Sort by a negative ratio of saltiness to sweetness, then if equal by saltiness
    # (To minimize taste ratios when accumulated)
    dishes.sort(key=lambda x: (X / (x[0] + 1e-10), x[1]), reverse=True)
    
    # Accumulate sweetness and saltiness
    total_sweetness = list(accumulate([dish[0] for dish in dishes]))
    total_saltiness = list(accumulate([dish[1] for dish in dishes]))
    
    # Binary search on the first position where either sweetness or saltiness exceeds limits
    limit_reached_at = min(
        bisect_left(total_sweetness, X),
        bisect_left(total_saltiness, Y)
    )
    
    return limit_reached_at

N, X, Y = map(int, input().split())
dishes = [tuple(map(int, input().split())) for _ in range(N)]

result = solve(N, X, Y, dishes)
print(result)