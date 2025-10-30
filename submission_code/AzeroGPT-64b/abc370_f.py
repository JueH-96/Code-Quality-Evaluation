import sys
from typing import List

def find_min_mass_and_uncut_lines(N: int, K: int, masses: List[int]) -> List[int]:
    """
    Finds the minimum mass that can be given to each of K people by dividing a cake
    into consecutive pieces and the number of cut lines that are never cut.
    
    :param N: Number of pieces the cake is divided into.
    :param K: Number of people to divide the cake among.
    :param masses: List of masses of each piece of the cake.
    :return: [minimum mass given to each person, number of cut lines never cut]
    """
    def can_divide(m: int, k: int) -> bool:
        count, current = 1, 0
        for i in range(2 * N):
            current += sums[i % N]
            if current >= m:
                current = 0 if i % N == N - 1 else sums[(i + 1) % N]
                count += 1
                if count == k + 1:
                    return True
        return False

    sums = masses + masses
    total_mass = sum(masses)
    low, high = -1, total_mass

    while high - low > 1:
        mid = (low + high) // 2
        if can_divide(mid, K):
            low = mid
        else:
            high = mid

    min_mass = low
    no_cut_lines = N - sum([1 for i in range(N - 1) if can_divide(min_mass, K) and can_divide(min_mass, K + 1, i + 1)])
    
    return [min_mass, no_cut_lines]

# Read input
N, K = map(int, input().split())
masses = list(map(int, input().split()))

# Solve and print output
result = find_min_mass_and_uncut_lines(N, K, masses)
print(*result)