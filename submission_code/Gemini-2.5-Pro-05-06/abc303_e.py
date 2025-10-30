import sys
import collections

def solve():
    N = int(sys.stdin.readline())

    # degrees_list will store degree of vertex i at index i. Using 1-based indexing.
    degrees_list = [0] * (N + 1) 

    for _ in range(N - 1):
        u, v = map(int, sys.stdin.readline().split())
        degrees_list[u] += 1
        degrees_list[v] += 1

    N1_count = 0  # Count of vertices with degree 1
    V2_count = 0  # Count of vertices with degree 2
    
    # C_gt2_levels stores levels of stars whose centers had degree > 2
    C_gt2_levels = [] 

    for i in range(1, N + 1): # Iterate through vertices 1 to N
        deg = degrees_list[i]
        if deg == 1:
            N1_count += 1
        elif deg == 2:
            V2_count += 1
        elif deg > 2:
            C_gt2_levels.append(deg)
        # deg == 0 is not possible for a connected graph with N >= 3.
    
    # Calculate M, the number of initial stars.
    # Based on derivation: 3*M = N - N1_count + 2
    # The problem implies M_numerator will be divisible by 3.
    M_numerator = N - N1_count + 2
    M = M_numerator // 3

    all_levels = []
    all_levels.extend(C_gt2_levels)

    # Calculate num_level_2_stars (M_2)
    # This is V2_count - (number of initial leaves that became degree-2 vertices)
    # Number of added edges = M - 1
    # Number of initial leaves that became degree-2 = 2 * (M - 1)
    num_added_edges = M - 1
    # num_level_2_stars must be non-negative.
    # It can be shown num_level_2_stars = M - len(C_gt2_levels).
    num_level_2_stars = V2_count - 2 * num_added_edges
    
    for _ in range(num_level_2_stars):
        all_levels.append(2)
        
    all_levels.sort()
    
    print(*(all_levels))

if __name__ == '__main__':
    solve()