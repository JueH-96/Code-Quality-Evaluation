def solve():
    import sys
    sys.setrecursionlimit(10**7)

    input_data = sys.stdin.read().strip().split()
    N, M = map(int, input_data[:2])
    edges = input_data[2:]

    # ------------------------------------------------------------
    # 1) Parse input and build adjacency information
    # ------------------------------------------------------------
    adj = [[] for _ in range(N)]
    deg = [0]*N
    idx = 0
    for _ in range(M):
        u = int(edges[idx]); v = int(edges[idx+1])
        idx += 2
        u -= 1  # make 0-based
        v -= 1
        adj[u].append(v)
        adj[v].append(u)
        deg[u]+=1
        deg[v]+=1

    # Vertices with degree 0 have no constraints; we can assign them any positive integer.
    # We'll assign them 1 at the end if a solution for the rest exists.

    # S = list of vertices that have degree > 0
    S = [i for i in range(N) if deg[i] > 0]
    if len(S) == 0:
        # No edges => all vertices are isolated => any positive assignment works
        print("Yes")
        print(" ".join(["1"]*N))
        return

    # We will solve the system of linear equations (over GF(2)) that arises from:
    #   For each v in S:  XOR of X[u] for u in adj[v] = 0
    # Each X[v] is a bit (0 or 1) in the solution vector for that connected component.
    #
    # Once we have a solution space (of dimension d >= 1), we need to ensure that
    # no vertex in S is forced to 0 in every solution. If a vertex i in S is forced to 0,
    # then we cannot assign it a positive integer. We check that by attempting to set x_i=1
    # and see if that is solvable. If it's not solvable for some i, we must print "No".
    #
    # If for every i in S there is at least one solution T_i with T_i[i]=1, we gather
    # those solutions T_i. Then we pick a small subset of them (via a simple greedy set cover)
    # so that every j in S appears in at least one T_i's support. We assign one output bit
    # per chosen solution. In that bit-position, an entire solution T_i is used. Thus all
    # adjacency constraints are satisfied bit-by-bit (each T_i is in the solution space),
    # and each vertex in S gets at least one bit set → X_j != 0. For vertices not in S,
    # we simply assign 1.
    #
    # The system size is at most |S| <= 60, which is small enough to do a bit-based
    # Gaussian elimination in O(|S|^3) = O(60^3) comfortably.

    # ------------------------------------------------------------
    # 2) Prepare a GF(2) system of equations
    #    We'll store them as row equations: sum_{u in adj[v]} x[u] = 0 (mod 2), for v in S.
    #    The variables are x[0..|S|-1], corresponding to each vertex in S.
    # ------------------------------------------------------------

    # Map vertices in S to a compact index 0..|S|-1
    pos_in_s = {}
    for i, v in enumerate(S):
        pos_in_s[v] = i

    # We will have |S| equations, one per vertex in S.
    # Each equation is represented as a bitmask of length |S| (up to 60 bits),
    # so we can store it in a single 64-bit integer in Python.
    equations = [0]*len(S)  # row i corresponds to "sum of x[u_in_S] for u in adj(S[i]) = 0"

    for i, v in enumerate(S):
        mask = 0
        for nbr in adj[v]:
            if deg[nbr] > 0:  # Only consider neighbors that are in S
                mask ^= (1 << pos_in_s[nbr])
        equations[i] = mask

    # ------------------------------------------------------------
    # 3) Build a routine for GF(2) rank / basis extraction
    #    We'll do row-reduction on "equations". Each row is up to 60 bits.
    # ------------------------------------------------------------
    def gaussian_elimination_xor(rows):
        """
        Perform in-place Gaussian elimination on a list of bitmasks (rows).
        Returns:
          rank, and the row-echelon form in rows (modified).
          Also returns pivot_positions: an array of length rank,
          where pivot_positions[r] is the bit position of the pivot in row r.
        """
        # We have up to len(rows) equations, each up to 60 bits.
        rank = 0
        pivot_positions = []
        bit_len = len(rows)  # up to 60 rows, but also up to 60 bits
        for bitpos in reversed(range(bit_len)):  # from highest bit to lowest
            # find a row from rank..end that has bitpos set
            pivot_row = -1
            for r in range(rank, len(rows)):
                if (rows[r] >> bitpos) & 1:
                    pivot_row = r
                    break
            if pivot_row == -1:
                continue  # no pivot in this bitpos
            # swap pivot_row with row rank
            if pivot_row != rank:
                rows[rank], rows[pivot_row] = rows[pivot_row], rows[rank]
            # now eliminate
            piv = rows[rank]
            for r in range(len(rows)):
                if r != rank and ((rows[r] >> bitpos) & 1):
                    rows[r] ^= piv
            pivot_positions.append(bitpos)
            rank += 1
            if rank == bit_len:  # cannot exceed number of rows in any case
                break
        return rank, pivot_positions

    # Make a copy for rank check
    eq_copy = equations[:]
    rankA, pivots = gaussian_elimination_xor(eq_copy)
    # dimension of solution space = number_of_variables - rank
    # number_of_variables is len(S).
    dim = len(S) - rankA

    # If dim == 0, the only solution is the all-zero solution for these variables.
    # That means all x[v] = 0 for v in S, which is NOT allowed because those vertices
    # must get a positive integer. So if S is non-empty and dim==0, answer is "No".
    if dim == 0:
        # This means all x[S] are forced to 0
        print("No")
        return

    # ------------------------------------------------------------
    # 4) We want to see if each vertex i in S is "forced to 0".
    #    That is, for i in S, check if there's any solution with x[i]=1.
    #    If no, we must print "No."
    #    If yes, store that solution in T_i (60-bit mask).
    #
    #    We'll do this by re-doing row-reduction with a particular variable forced to 1.
    # ------------------------------------------------------------

    # We'll create a function that, given the row-echelon form, tries to solve for
    # x[idx_to_set] = 1 (where idx_to_set is in [0..|S|-1]) if possible, otherwise returns None.
    # If successful, it returns a 60-bit mask describing x in the order [0..|S|-1].
    #
    # Steps to do that:
    #   Let 'variables' = x[0..|S|-1].
    #   We have rankA, eq_copy (row echelon form), pivot positions, etc.
    #   Forcing x[idx_to_set] = 1 means we set that bit. Then we try to see if
    #   it can satisfy all equations eq_copy. If contradictory, no solution; else produce one.

    def build_row_echelon(equations):
        # Return a copy of equations in row echelon form along with pivot info
        eq_c = equations[:]
        r, p = gaussian_elimination_xor(eq_c)
        return eq_c, r, p

    echelon, rankA, pivot_positions = build_row_echelon(equations)

    # We'll do back-substitution in GF(2):
    # The row echelon form has 'rankA' pivot rows. pivot_positions[r] is the bit
    # index of the pivot in row r (in descending order).
    # The row 'r' has a pivot at pivot_positions[r].
    # That row says x[pivot_positions[r]] = ... sum of other bits in the row.
    # We'll process from top row to bottom row to solve for x (any free bits remain free).
    #
    # To force x[idx_to_set] = 1, we simply set that bit in our "partial solution" and see
    # if we can remain consistent. Then we back-substitute. If there's a contradiction,
    # no solution.

    def solve_forced_one(echelon, rankA, pivot_pos, idx_to_set):
        # We have up to len(S) variables (bits 0..|S|-1).
        # The echelon is from highest bit to lowest in some order, but we also have pivot_pos
        # in descending order. We'll keep a solution vector "sol" of length S bits, init to 0.
        sol = 0
        # Force x[idx_to_set]=1
        sol ^= (1 << idx_to_set)

        # We'll interpret echelon row r as:
        #   let pbit = pivot_pos[r],
        #   then row r (call it row_mask) has pbit set, and possibly other bits.
        #   We want sum_of row_mask's bits in 'sol' to be 0. i.e. dot(row_mask, sol)=0 mod2
        #   Because each row stands for sum_{bit in row_mask} x[bit] = 0.
        # We can fix bits in 'sol' to ensure that pivot bits unify that condition.

        for r in range(rankA):
            pbit = pivot_pos[r]
            row_mask = echelon[r]
            # The sum of bits in sol that appear in row_mask must be 0 mod 2.
            # We check what it currently is:
            current_sum = bin(sol & row_mask).count("1") % 2
            if current_sum != 0:
                # We want to flip one bit in sol (which is a free bit in row_mask besides pbit),
                # but typically we'd choose the pivot bit. However, the pivot bit is part of row_mask
                # but we want that bit to be determined by the row, not free.
                #
                # In normal back-substitution, we'd solve for x[pbit] from the bits > pbit
                # or something. But here let's do a direct approach:
                # If the row_mask has other bits besides pbit, we can flip them, or if pbit is free,
                # we can flip pbit. But usually pivot means pbit depends on other free bits.
                #
                # A simpler approach is: if there's some other bit 'fb' in row_mask (fb != pbit),
                # we can flip that. But in row echelon form, typically all bits in row_mask that
                # are higher than pbit have been eliminated, so the only unknown might be pbit itself.
                # Actually let's do a simpler known technique:
                #   If the row is "pbit ^ (other pivot bits) = 0", we solve for pbit = sum of other bits in row (besides pbit).
                # We'll do a standard forward/back approach. We can do it from top to bottom or bottom to top.
                #
                # Instead, let's realize if the system is consistent, we can fix free bits as needed.
                # But we forced one bit. Possibly we caused an inconsistency. Then there's no solution.
                #
                # Let's do a standard check: If row_mask only has pbit (i.e. it's exactly 1<<pbit),
                # then we must have sol[pbit] = 0 (contradiction if it's currently 1).
                # If row_mask has other bits, we might fix one of those to restore the sum to 0.
                #
                # Implementation detail: If the row_mask == (1<<pbit), then that row says x[pbit] = 0.
                # If we have sol[pbit] = 1, contradiction -> no solution.
                # If there's more bits, we can try flipping one of them. But to do that systematically,
                # we can proceed from lower pivot to higher pivot (or vice versa).
                # Let's do a quick pivot-check approach:

                if row_mask == (1 << pbit):
                    # This demands x[pbit]=0, but we have it = 1 => contradiction
                    return None
                else:
                    # There must be some other bit in row_mask that we can flip
                    # because pbit is the "lowest" in that row_mask. Let's find any free bit in row_mask
                    # that is < pbit. We'll flip it in sol.
                    # But if there is no such bit, contradiction. Usually row echelon form means
                    # that row_mask's highest set-bit is pbit, but there could be others below pbit.
                    other_mask = row_mask ^ (1 << pbit)
                    if other_mask == 0:
                        # no other bits => contradiction
                        return None
                    # pick the lowest bit in other_mask to flip
                    free_bit = (other_mask & (-other_mask))  # isolate lowest set bit
                    sol ^= free_bit  # flip it, adjusting our solution so that sum becomes 0 now
                    # now the sum in that row is fixed to 0

        # After that pass, we verify again if all rows sum to 0.
        for r in range(rankA):
            row_mask = echelon[r]
            s = bin(sol & row_mask).count("1") % 2
            if s != 0:
                # if still not 0, no solution
                return None

        return sol

    # We'll attempt to compute a solution T_i for each i in S with x[i]=1.
    # If it's impossible, we say "No". If possible, store it.
    T = [0]*len(S)  # T[i] will be a 60-bit mask describing a solution that has
                    # x[i] = 1 and satisfies all adjacency constraints (over GF(2)).
    for i in range(len(S)):
        sol_i = solve_forced_one(echelon, rankA, pivot_positions, i)
        if sol_i is None:
            # that means i is forced to 0, no solution with x[i]=1
            print("No")
            return
        T[i] = sol_i

    # ------------------------------------------------------------
    # 5) Now we have for each i in S a solution T_i whose support includes i.
    #    We want to cover all vertices j in S (i.e. for each j in S, we want them
    #    to appear in at least one T_i's support), so that in the final assignment
    #    each j gets at least one bit set → X_j != 0.
    #
    #    If any j in S is not in the support of at least one T_i, that means j is forced to 0
    #    in all solutions with a single 'forced bit' → no solution overall. Then "No".
    #
    #    Otherwise, we do a simple greedy set cover using these T_i. We'll choose up to
    #    len(S) of them (which is <= 60) to cover all S. Then we assign those chosen sets
    #    to distinct bit-positions in the final integer assignment.
    # ------------------------------------------------------------
    uncovered = set(range(len(S)))
    sets_indices = []  # which i in S we choose
    chosen_bitmasks = []
    while uncovered and len(sets_indices) < len(S):
        # pick T_i that covers the most uncovered vertices
        best_i = None
        best_cover = 0
        best_cover_set = None
        for i in range(len(S)):
            cov = 0
            cset = []
            mask = T[i]
            # count how many from 'uncovered' are in T[i]'s support
            # support of T[i] = indices where (mask >> idx) & 1 = 1
            tmp = mask
            # We'll do a quick iteration
            # But S can be up to 60, so let's just do a small loop.
            for v in uncovered:
                if ((tmp >> v) & 1) == 1:
                    cov += 1
            if cov > best_cover:
                best_cover = cov
                best_i = i
        if best_i is None or best_cover == 0:
            # That means we can't cover any new vertex => fail
            break
        sets_indices.append(best_i)
        # remove covered
        chosen_mask = T[best_i]
        to_remove = []
        for v in uncovered:
            if ((chosen_mask >> v) & 1) == 1:
                to_remove.append(v)
        for vv in to_remove:
            uncovered.remove(vv)

    if uncovered:
        # We failed to cover all vertices in S
        # But let's check if there's maybe a smaller dimension situation...
        # Actually if any vertex j in S wasn't covered by any T_i, that means it never appears
        # with bit=1 in any single-forced solution -> forced to 0 => No
        print("No")
        return

    # Great, we have a small set sets_indices. We'll use one distinct bit for each chosen T_i.
    # We'll build a final bitmask array F (length = number_of_chosen_sets), each is T_{best_i}.
    # Then for each vertex j in S, its final assignment X_j will be the OR of bits in which
    # j is in the support of T_i.
    # Example: if sets_indices = [i1, i2, ...], then bit b in X_j is 1 if (T_i(b) >> j)&1 ==1,
    # and i(b) is the index in sets_indices for that bit.
    # We'll actually do: final_X[j] = sum_{b=0..k-1} ( ((T_i(b) >> j)&1 ) << b ).
    # We'll store them in a list finalX of length |S|. Also for deg=0 vertices, we assign 1.

    finalX = [0]*N
    # Build the chosen masks
    chosen_bitmasks = [T[i] for i in sets_indices]
    # For each j in S, compute X_j
    for j_index, j in enumerate(S):
        val = 0
        for b_index, mask in enumerate(chosen_bitmasks):
            if ((mask >> j_index) & 1) == 1:
                # set bit b_index
                val |= (1 << b_index)
        if val == 0:
            # that means j is not covered? Shouldn't happen
            print("No")
            return
        finalX[j] = val

    # For vertices with deg=0, assign 1
    for i in range(N):
        if deg[i] == 0:
            finalX[i] = 1

    # Now we have an assignment in range [1..(1<<len(S)) -1] for those in S,
    # and 1 for deg=0 vertices. This is certainly ≤ 2^60 -1, since |S| ≤ 60.
    # Check if all are in [1..2^60-1].
    # (They should be, as none is 0, and they are at most 2^60-1 if there are at most 60 bits used.)
    for i in range(N):
        if finalX[i] == 0 or finalX[i] >= (1 << 60):
            print("No")
            return

    # ------------------------------------------------------------
    # 6) Print the result
    # ------------------------------------------------------------
    print("Yes")
    print(" ".join(str(x) for x in finalX))


def _test():
    # Provided samples
    import io
    import sys

    input_data = """3 3
1 2
1 3
2 3
"""
    print(run_io_fun(input_data, solve))

    input_data = """2 1
1 2
"""
    print(run_io_fun(input_data, solve))

    input_data = """1 0
"""
    print(run_io_fun(input_data, solve))

    input_data = """4 5
1 2
1 3
2 3
2 4
3 4
"""
    print(run_io_fun(input_data, solve))

def run_io_fun(input_data, func):
    import sys
    backup_stdin = sys.stdin
    backup_stdout = sys.stdout
    sys.stdin = io.StringIO(input_data)
    output_buf = io.StringIO()
    sys.stdout = output_buf
    func()
    sys.stdin = backup_stdin
    sys.stdout = backup_stdout
    return output_buf.getvalue()

# If you want to run local tests, you can uncomment _test() call below.
# However, the final submission should only contain solve() followed by solve() call if desired.
# _test()

# The problem statement asks to define solve() and then call it.
# We'll just trust that the online judge will call solve().
# If needed, we can call solve() here, but typically "solve()" alone is enough.
# solve()