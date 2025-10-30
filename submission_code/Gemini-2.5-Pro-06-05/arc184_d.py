import sys

# It is necessary to increase the recursion limit for deep recursion.
# N can be up to 300, and a chain of operations could lead to a recursion depth of N.
sys.setrecursionlimit(2000)

MOD = 998244353

def solve():
    """
    Main function to solve the problem.
    Reads input, sets up the recursive calculation, and prints the result.
    """
    try:
        N_str = sys.stdin.readline()
        if not N_str: return
        N = int(N_str)
        
        if N == 0:
            print(1) # The only set is the empty set
            return
            
        points_input = [tuple(map(int, sys.stdin.readline().split())) for _ in range(N)]
    except (IOError, ValueError):
        return

    # Sort points by X-coordinate. This simplifies quadrant checks.
    points = sorted(points_input)

    # Memoization table for our recursive function.
    # The key will be a tuple of indices of points in a subproblem.
    memo = {}

    def count_sets(p_indices_tuple):
        """
        Recursively calculates the number of possible sets for a given subset of points.
        
        Args:
            p_indices_tuple: A tuple of indices into the sorted `points` array.
                             The tuple must be sorted for canonical memoization keys.
        
        Returns:
            The number of possible sets, modulo MOD.
        """
        if not p_indices_tuple:
            # Base case: an empty set of points can only result in one set (the empty set).
            return 1

        if p_indices_tuple in memo:
            return memo[p_indices_tuple]

        # The current set of points itself is one possible outcome (0 operations).
        total = 1
        
        for p_k_idx in p_indices_tuple:
            # p_k_idx is the index in the global `points` array.
            
            x_k, y_k = points[p_k_idx]
            
            # Check if picking p_k_idx as a pivot changes the set.
            is_changed = False
            for p_i_idx in p_indices_tuple:
                if p_i_idx == p_k_idx:
                    continue
                
                x_i, y_i = points[p_i_idx]
                
                # The operation removes points in SW and NE quadrants relative to pivot k.
                # Points are sorted by X, so p_i_idx < p_k_idx implies x_i < x_k, etc.
                if (p_i_idx < p_k_idx and y_i < y_k) or \
                   (p_i_idx > p_k_idx and y_i > y_k):
                    is_changed = True
                    break
            
            if is_changed:
                L_k = [] # Points in the NW quadrant
                R_k = [] # Points in the SE quadrant
                
                for p_i_idx in p_indices_tuple:
                    if p_i_idx == p_k_idx:
                        continue
                    
                    x_i, y_i = points[p_i_idx]

                    if p_i_idx < p_k_idx: # x_i < x_k
                        if y_i > y_k:
                            L_k.append(p_i_idx)
                    else: # p_i_idx > p_k_idx, so x_i > x_k
                        if y_i < y_k:
                            R_k.append(p_i_idx)
                
                # The recursive calls are on tuples. Indices in L_k and R_k are
                # already sorted because they are sub-sequences of p_indices_tuple.
                term = (count_sets(tuple(L_k)) * count_sets(tuple(R_k))) % MOD
                total = (total + term) % MOD

        memo[p_indices_tuple] = total
        return total

    # The initial state is the set of all N points.
    initial_indices = tuple(range(N))
    result = count_sets(initial_indices)
    print(result)

solve()