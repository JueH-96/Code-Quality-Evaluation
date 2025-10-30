import sys
from collections import defaultdict

def solve():
    N = int(sys.stdin.readline())

    # No need for adjacency list in this specific solution,
    # just degrees are sufficient.
    degree = [0] * (N + 1)

    for _ in range(N - 1):
        u, v = map(int, sys.stdin.readline().split())
        degree[u] += 1
        degree[v] += 1

    num_deg1 = 0
    for i in range(1, N + 1):
        if degree[i] == 1:
            num_deg1 += 1

    # Formula derivation:
    # Let S_L = sum of levels (L_1 + ... + L_M)
    # Let M = number of initial stars
    # Total vertices N = sum of (vertices per star) = sum(L_i + 1) = S_L + M
    # Total edges in final tree T = N-1
    # Total edges within initial stars = S_L
    # Number of edges added by Takahashi = E_added = (N-1) - S_L
    # Each added edge connects two original leaf vertices. These leaves become degree 2 in T.
    # The remaining original leaves (those not involved in an added edge) stay degree 1 in T.
    # So, num_deg1 = (total original leaves) - (2 * number of added edges)
    # num_deg1 = S_L - 2 * E_added
    # Substitute E_added: num_deg1 = S_L - 2 * ((N-1) - S_L)
    # num_deg1 = S_L - 2N + 2 + 2S_L
    # num_deg1 = 3S_L - 2N + 2
    # Rearranging for S_L:
    # 3S_L = num_deg1 + 2N - 2
    # S_L = (num_deg1 + 2N - 2) / 3
    
    total_sum_of_levels = (num_deg1 + 2 * N - 2) // 3

    star_levels_found = []
    
    # Vertices with degree > 2 in T must be central nodes of stars.
    # Their degree directly represents the level of that star.
    for i in range(1, N + 1):
        if degree[i] > 2:
            star_levels_found.append(degree[i])
    
    # Calculate the sum of levels already identified.
    current_sum_of_collected_levels = sum(star_levels_found)
    
    # The remaining sum of levels must come from level-2 stars.
    # This is because other nodes (with degree 1 or 2) are either leaves or
    # central nodes of level-2 stars. Since k >= 2, any central node not
    # already identified must be of a level-2 star.
    remaining_sum_of_levels = total_sum_of_levels - current_sum_of_collected_levels
    
    # Each level-2 star contributes 2 to the sum of levels.
    # So, the number of level-2 stars is remaining_sum_of_levels / 2.
    num_level_2_stars = remaining_sum_of_levels // 2
    
    for _ in range(num_level_2_stars):
        star_levels_found.append(2)

    star_levels_found.sort()
    
    # Print the sorted levels, space-separated.
    print(*(star_levels_found))

solve()