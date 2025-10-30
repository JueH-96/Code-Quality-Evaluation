import sys

def solve():
    """
    Solves the graph game problem by analyzing the structure of the game.
    """
    # It's recommended to run this with PyPy for performance on large inputs.
    sys.setrecursionlimit(2 * 10**5 + 5)
    
    try:
        input = sys.stdin.readline
        N, M = map(int, input().split())
        adj = [[] for _ in range(N)]
        for _ in range(M):
            u, v = map(int, input().split())
            u -= 1
            v -= 1
            adj[u].append(v)
            adj[v].append(u)
    except (IOError, ValueError):
        # Handle potential empty input on some platforms
        return

    visited = [False] * N
    
    k = 0  # Number of connected components
    k_odd = 0  # Number of components with an odd number of vertices
    
    # In case 3, we need sum of s_i * t_i for all components
    sum_si_ti = 0

    # In case 2, we need the number of components with (odd, odd) partitions
    num_oo_if_ko_zero = 0

    for i in range(N):
        if not visited[i]:
            k += 1
            q = [i]
            visited[i] = True
            
            # Bipartition coloring
            colors = {i: 0}
            s = 1  # Count for color 0
            t = 0  # Count for color 1

            head = 0
            while head < len(q):
                u = q[head]
                head += 1
                
                for v in adj[u]:
                    if v not in colors:
                        colors[v] = 1 - colors[u]
                        if colors[v] == 0:
                            s += 1
                        else:
                            t += 1
                        visited[v] = True
                        q.append(v)
            
            sum_si_ti += s * t
            
            component_size = s + t
            if component_size % 2 == 1:
                k_odd += 1
            else:
                if s % 2 == 1:  # and t % 2 == 1
                    num_oo_if_ko_zero += 1

    if k_odd % 2 == 1:
        # Implies N is odd. S*T is always even.
        # Winner is determined by parity of M.
        if M % 2 == 1:
            print("Aoki")
        else:
            print("Takahashi")
    elif k_odd == 0:
        # All components have even size, so N is even.
        # The parity of the final |S|*|T| is fixed.
        # It's odd iff num_oo is odd.
        # Winner determined by parity of (|S|*|T| - M).
        if (num_oo_if_ko_zero - M) % 2 == 1:
            print("Aoki")
        else:
            print("Takahashi")
    else:  # k_odd is even and > 0
        # N is even. A choice of final partition parity exists.
        # This is the complex case solvable with a Nim-sum-like argument.
        q_choices = k_odd // 2
        
        # Number of moves available within initial components
        l_moves = sum_si_ti - M
        
        # Number of non-choice-granting connection moves
        other_conn_moves = k - 1 - q_choices
        
        num_normal_moves = l_moves + other_conn_moves
        
        # g_choice = 1 (since q > 0), g_normal = num_normal_moves % 2
        # Aoki wins if g_choice ^ g_normal != 0, i.e., 1 ^ (num_normal_moves % 2) != 0
        # This is true if num_normal_moves % 2 == 0
        if num_normal_moves % 2 == 0:
            print("Aoki")
        else:
            print("Takahashi")

solve()