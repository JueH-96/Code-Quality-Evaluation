def main():
    import sys
    sys.setrecursionlimit(10**7)

    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    M = int(input_data[1])
    edges = input_data[2:]

    # Edge case: no edges => any assignment (e.g. all 1's) is fine
    if M == 0:
        # Print "Yes" and then all 1's
        print("Yes")
        print(" ".join(["1"]*N))
        return

    # Build adjacency list (0-based)
    adj = [[] for _ in range(N)]
    deg = [0]*N
    idx = 0
    for _ in range(M):
        u = int(edges[idx]) - 1
        v = int(edges[idx+1]) - 1
        idx += 2
        adj[u].append(v)
        adj[v].append(u)
        deg[u] += 1
        deg[v] += 1

    # If any vertex has degree 1, answer is No (reasoning: that forces
    # its single neighbor's value to be 0 in XOR, but 0 is not allowed).
    for d in deg:
        if d == 1:
            print("No")
            return

    # We will solve the system "for each v, XOR_of(adj[v]) = 0".
    # In GF(2)-linear algebra terms, for each v:
    #      sum_{w in adj[v]} x_w = 0.
    #
    # This means M_matrix[v, w] = 1 if w in adj[v], else 0,
    # and we want M_matrix * X = 0 in GF(2).
    #
    # Solve connected component by connected component.  For each component that
    # actually has edges, we find a basis for its null space.  Then we must
    # ensure that each vertex in that component can be made nonzero in at least
    # one chosen combination of basis vectors (across up to 60 bits).  If there
    # is a vertex that is zero in every null-space vector, then there's no way
    # to assign that vertex a nonzero integer => "No".
    #
    # Implementation plan:
    # 1) Find connected components.
    # 2) For each component that has edges, construct the matrix M_comp,
    #    find a null-space basis B_comp (dimension >= 1 since there's at least
    #    one cycle), check coverage (if a vertex is 0 in all basis vectors,
    #    output No). Otherwise, pick enough basis vectors (up to 60) to cover
    #    all vertices so each vertex has at least one bit = 1 in the final.
    # 3) Combine into a global solution array X of length N.
    #    - Isolated vertices can be assigned 1 trivially.
    #
    # We'll do a standard GF(2) nullspace finder (via row-echelon or by
    # building the matrix for M*x=0, but be mindful that M could be up to NxN
    # with N up to 60 => we can do bit-based elimination.

    # Build an undirected graph for components
    visited = [False]*N
    graph_adj = [[] for _ in range(N)]
    for i in range(N):
        for w in adj[i]:
            graph_adj[i].append(w)

    def dfs_collect(start):
        stack = [start]
        comp = []
        visited[start] = True
        while stack:
            node = stack.pop()
            comp.append(node)
            for nx in graph_adj[node]:
                if not visited[nx]:
                    visited[nx] = True
                    stack.append(nx)
        return comp

    comps = []
    for i in range(N):
        if not visited[i]:
            comps.append(dfs_collect(i))

    # We will store the final 60-bit assignment in an array of Python ints:
    X = [0]*N

    # Function to find the null space basis (over GF(2)) of the "adjacency-rows"
    # restricted to a given component.  The matrix M_comp has "len_comp" rows
    # (one per vertex in the component) and "len_comp" columns (same vertices),
    # where M_comp[r,c] = 1 if (component[r], component[c]) is an edge,
    # else 0.  Then we want M_comp * x = 0 in GF(2).
    # We'll return a list of basis vectors, each vector is a length-len_comp
    # array/list of bits (0 or 1).  Actually we can store each vector as an
    # integer bitmask up to len_comp <= 60.

    def find_nullspace_basis(comp_vertices):
        """
        Return a list of bitmasks, each up to 1<<len(comp_vertices),
        forming a basis of solutions x in GF(2) to M*x=0, where
        M[r,c] = 1 if there's an edge between comp_vertices[r] and comp_vertices[c],
        else 0.  Each row r imposes sum_{c in neighbors(r)} x_c = 0 in GF(2).
        Number of rows = number of columns = len_comp.
        """
        csize = len(comp_vertices)
        # Build the matrix M (csize x csize). M[r] as one integer bitmask for columns
        Mrows = [0]*csize
        # row r corresponds to vertex comp_vertices[r]
        # column c also corresponds to comp_vertices[c]
        # set bit c of Mrows[r] if adjacency[r] includes comp_vertices[c].
        idx_of = {}
        for i, v in enumerate(comp_vertices):
            idx_of[v] = i

        for r in range(csize):
            v_r = comp_vertices[r]
            mask = 0
            for w in adj[v_r]:
                if w in idx_of:  # same component
                    c = idx_of[w]
                    mask |= (1 << c)
            Mrows[r] = mask

        # We want solutions x in GF(2)^csize to M*x=0 where row r => popcount(M[r] & x) %2=0
        # We'll transpose viewpoint: we want to find x in the nullspace.
        # But it's simpler to treat Mrows[r] as the row. We'll do row echelon on Mrows.
        # Then the pivot positions define constraints on x. We'll collect a basis for
        # the solution space. Actually it's easier to build the matrix for columns
        # or do standard "XOR-based" Gaussian elimination, but we must be careful
        # to interpret it correctly. We want M*x=0, i.e. for each row r,
        # dot(M[r], x)=0 in GF(2). If M has rank rnk, dimension of nullspace is csize-rnk.

        # We'll do a standard technique for computing the nullspace of M by
        # augmenting with the identity matrix and performing row ops. But
        # note we only need the basis of the nullspace, which can be found by
        # standard pivoting.

        # Let's build an array of length csize of bitmasks of length (csize + csize):
        # left csize bits = M row, right csize bits = identity row. Then do elimination
        # on the left csize portion. At the end, the free columns in the left side
        # give basis vectors from the right side. This is a classic approach, but
        # we must be mindful that csize <= 60 => we can store 2*csize in 120 bits,
        # which fits in Python's int (unbounded). So let's do so.

        extended = []
        for r in range(csize):
            left_part = Mrows[r]
            # right_part is identity => 1 << r
            right_part = 1 << r
            row_full = (left_part << csize) ^ right_part
            extended.append(row_full)

        # Now do elimination on the left half (bits from 2*csize-1 down to csize).
        # We'll keep track of which column is pivot. We'll do up to csize columns.
        pivot_col = [-1]*csize  # pivot_col[row] = which col is pivot for that row
        col_used = [False]*csize

        row_i = 0
        for col in range(csize):
            # Try to find pivot in row>=row_i with that col bit = 1
            mask_col_bit = 1 << (csize-1 - col)  # if csize=4, col=0 => we want bit 1<<(3)
            # Actually let's clarify which bit in the left side corresponds to column col:
            # The left side occupies bits [csize..2*csize-1].  We'll define the top-left bit
            # as col=0 => bit index (2*csize-1). So the bit for column col is
            # bit_index = 2*csize-1 - col.
            bit_index = (2*csize-1) - col
            pivot_found = -1
            for r in range(row_i, csize):
                if ((extended[r] >> bit_index) & 1) == 1:
                    pivot_found = r
                    break
            if pivot_found < 0:
                # no pivot in this column
                continue
            # swap pivot_found with row_i
            if pivot_found != row_i:
                extended[row_i], extended[pivot_found] = extended[pivot_found], extended[row_i]
            pivot_col[row_i] = col
            col_used[col] = True

            # Eliminate downward
            pivot_val = extended[row_i]
            for rr in range(row_i+1, csize):
                if ((extended[rr] >> bit_index) & 1) == 1:
                    extended[rr] ^= pivot_val
            row_i += 1
            if row_i == csize:
                break

        rank_ = row_i

        # Now we have row-echelon form (not fully reduced). The free columns are those
        # where col_used[col] = False. Each free column cfree yields a basis vector:
        # Set that free col to 1, other free cols to 0, solve for pivot cols.
        # We'll proceed from bottom to top to fill pivot columns.
        basis = []
        free_cols = [c for c in range(csize) if not col_used[c]]

        # We'll convert the final row-echelon form for convenience:
        # do upward elimination so each pivot row is the only 1 in that pivot column
        # in the left half.
        for r in reversed(range(rank_)):
            pc = pivot_col[r]
            if pc == -1:
                continue
            bit_index = (2*csize-1) - pc
            # eliminate upwards
            for rr in range(r):
                if ((extended[rr] >> bit_index) & 1) == 1:
                    extended[rr] ^= extended[r]

        # Now read off pivot equations explicitly if needed. Then for each free col, we build
        # a solution vector.
        for cfree in free_cols:
            # We'll set that column to 1, others to 0, then read pivot columns from rows
            vec_mask = 0
            # set x[cfree] = 1
            # pivot x's are determined from that
            # pivot rows from top to bottom:
            # x[pivot_col[r]] = sum of row r's right half bits if the free col bit is set in row r
            # Actually easier to reconstruct by plugging into the row equations.
            # We'll build a temporary "solution array" of length csize in GF(2), store in integer.
            x_temp = 0
            x_temp |= (1 << (csize-1 - cfree))  # in GF(2), x_temp's indexing is reversed?
            # Actually let's keep it simpler: We'll store solution in normal left->right bit order,
            # i.e. the least significant bit is x[0], the next is x[1], etc.  We'll fill that at end.
            # For now, let's keep a small array of length csize:
            xarr = [0]*csize
            xarr[cfree] = 1

            # Now pivot rows from top to bottom:
            for r in range(rank_):
                pc = pivot_col[r]
                if pc == -1:
                    continue
                # row r says: left_part = 0 => sum_{c in row} x[c] = 0
                # but we've now got the row in form where the pivot col in row r
                # is the only 1 in that row's left side. So extended[r]'s right half
                # is the "solution offset" if the sum of the free columns in that row is 1.
                row_val = extended[r]
                # gather which free columns are 1 in that row's left side. Actually
                # in row echelon form, the row's left side has exactly 1 bit: the pivot col pc.
                # So x[pc] = sum of row's right side bits if there's any 1 among free col bits in row?
                # Actually simpler: if we consider the row's right half for the basis vector approach,
                # we set pivot x = the row's right side if the free col is used in that row. 
                # Let's just read off the row's entire right half:
                right_mask = row_val & ((1 << csize) - 1)  # lower csize bits
                # The pivot col's bit in the left half is definitely 1 for row r. We want to see
                # if cfree is included in that row. Actually we can see if row_val's left half
                # has the free col bit set. But in *reduced* form, it should or shouldn't. 
                # Easiest is to compute the sum of all known xarr[ c where left_half_bit=1 except pc ],
                # but that set is empty if each pivot row has exactly that pivot bit = 1. So the row eq
                # becomes xarr[pc] = sum_of_row_right_half_in_GF2
                # Because the row is now in a form where the pivot col is the only left bit set in that row.
                # So xarr[pc] = the parity of bits in right_mask that correspond to xarr. Actually we must
                # interpret the right_mask as the partial solution for that row. But the row echelon
                # form's upward elimination should have zeroed out everything above, so we can read it off:
                # the pivot variable = sum of the row's right side bits where xarr bit is 1 in that row's
                # left side. Now there's no other left bit in that row, so xarr[pc] = the parity of that
                # row's right half if the row's left side has indeed the free col cfree set? Actually let's
                # do a direct approach: We look at the left side bits except pc. They should be 0 in this row
                # after upward elimination. So the row equation is x_pc + 0 = the row's right side as bits
                # of x (?). Let me simplify with a direct read:
                
                # The row's right side is the bits [0..csize-1], with bit 0 as the LSB. We can see which
                # bits are set. Then xarr[pc] = sum of those bits in xarr[...] if the row's left side
                # has them set. But that set is only pc itself. So effectively, xarr[pc] = 0 if row's
                # right side is 0, else 1, for this basis vector? Actually we must see how the free col
                # cfree influences that row.
                
                # Rather than overcomplicate, let's just do the standard: M*x=0 => row r => pivot col p:
                # x[pc] = sum_{all other columns c' where left_bit=1} x[c'] plus the row's right bits...
                # But we've forced all other pivot columns to 0 in that row by upward elimination. So
                # x[pc] = the sum of the row's right half's bits if row's left half free bits are 1
                # because a free col cfree might appear in that row. Actually let's do a quick method:
                # We'll see if the row's left portion (excluding pc) had any free col set => but that
                # should now be 0 after upward elimination. So x[pc] = 0 plus (the bits from the row's
                # right portion). So we just see if the row's right portion is 1 in the cfree position's
                # row? 
                
                # Actually an easier method: we can apply the row's equation to the partial xarr. 
                # The row's left portion says sum_{col where bit left side=1} xarr[col] = 0
                # We already have xarr for free cols. The pivot col pc is unknown yet. So
                # xarr[pc] = sum_{col in left portion (excluding pc) if that bit=1} xarr[col]
                #           XOR sum_{bits in right portion} (but we must interpret the right portion
                #           as a block of csize bits representing the row's "offset"? Actually in
                #           standard AX=0, there's no constant offset. The right portion is
                #           "augmented" identity used for tracking transformations. So we want
                #           xarr[pc] = sum of those augmented bits that correspond to whichever free
                #           col we set to 1. 
                #
                # Let me do something simpler: We'll store the final extended matrix row r entirely.
                # The left half is the pivot col pc alone => bit at position (2*csize-1-pc) is 1.
                # The right half is the transformation from the original identity. If the cfree-th column
                # is set in that row's left portion, we XOR row's right portion into xarr. But after
                # upward elimination, it won't be set unless pc == cfree. In that case, we do xarr[pc]
                # = 1? Actually let's do a direct approach: Let me skip the typical factorization. We'll
                # do the standard "choose free col cfree => set it to 1. Then forward-substitute"
                # in the final row-echelon system from top row to bottom row:
                pass

            # We'll do the standard forward-substitution from top to bottom:
            # (We actually have the matrix in upper form, but let's do a neat approach.)
            # We'll re-run a simpler forward-sub with the row-echelon extended:
            # Set x[cfree] = 1, and x of other free columns = 0. Then for each pivot row in order:
            # the pivot col pc: x[pc] = sum of left side's known x minus the right side? But the
            # right side is the row's last csize bits. Actually it's simpler if we just store the
            # row echelon form in memory. Let's do it carefully:

            xarr = [0]*csize
            xarr[cfree] = 1
            for r in range(rank_):
                pc = pivot_col[r]
                if pc < 0:  # no pivot
                    continue
                # row r => left side = pivot col pc plus possibly other free col bits (which we set).
                # But after upward elimination, only one left bit remains for row r => pc. So
                # sum_{all columns cLeft where row's left side bit=1} xarr[cLeft]
                #   = sum_{bits in extended[r]'s right half} in GF(2).
                left_mask = (extended[r] >> csize)  # top csize bits
                # pivot bit for pc:
                pivot_bit = 1 << (csize-1 - pc)
                # we check which free cols are set in left_mask (besides pc). In ideal form, that
                # should be just pc. But let's just do a popcount = 0 or 1 from xarr to find sum_left.
                # We'll gather columns that are 1 in left_mask:
                # but typically there's exactly 1 => pc. Let's do it more general though:

                sum_left = 0
                col_idx = 0
                tmp_mask = left_mask
                while tmp_mask:
                    lowb = (tmp_mask & -tmp_mask)
                    bpos = (lowb).bit_length()-1  # position in 0-based from LSB
                    # that means column = ?
                    # if bpos=0 => that corresponds to col=0? Actually, we have to be careful.
                    # left_mask bit bpos => column c = bpos. But actually, the row's left half bits
                    # are stored in the top csize bits. We'll decode properly:
                    ccol = (csize-1) - bpos  # since bit 0 is the LSB => col= (csize-1 - bpos).
                    sum_left ^= xarr[ccol]
                    tmp_mask ^= lowb

                sum_right = 0
                # row's right half is extended[r] & ((1<<csize)-1)
                right_half = extended[r] & ((1 << csize) - 1)
                # We want sum_left = sum_right in GF(2). So if sum_left != sum_right, we must flip xarr[pc].
                if sum_left != (right_half % 2):  # if they differ in parity
                    xarr[pc] ^= 1

            # Now xarr is the solution in GF(2). We'll build a bitmask out of xarr:
            vec_mask = 0
            for i_bit in range(csize):
                if xarr[i_bit] == 1:
                    vec_mask |= (1 << i_bit)
            basis.append(vec_mask)

        # Additionally, if the dimension < (csize - rank_), we might have fewer free cols than that,
        # but that is OK. We should have exactly len(free_cols) basis vectors. That is our basis
        # for the nullspace (plus the trivial 0 vector).
        #
        # We might also want to check if rank_ < csize.  If rank_ == csize, the only solution is x=0,
        # which is invalid. But that can't happen if there's a cycle. We'll handle that anyway:
        if rank_ == csize:
            # means only trivial solution => no good
            return None  # indicates no nontrivial solution
        return basis

    # We'll assign each connected component's solution bits.  For each comp that has
    # at least one edge, we find its nullspace basis.  Then we pick a subset of those basis
    # vectors (up to 60 of them) so that every vertex in that component can be made nonzero
    # in the final combination.  We do this by a simple "set cover" approach in GF(2)-sense:
    #
    # We'll maintain a boolean "covered[v]" meaning vertex v already has at least one bit = 1
    # in the final assignment.  Initially all False.  We'll keep a list "selected_vectors = []".
    # While there's a vertex v not covered, pick any basis vector that has bit v=1, add it to
    # selected_vectors.  If we exceed 60 vectors but still not all covered => fail => "No".
    #
    # If there is a vertex that is 0 in every basis vector => impossible => "No".
    #
    # Once we have selected_vectors (say we pick k of them <= 60), we assign each one to a distinct
    # bit position in the final 60-bit integer for that component's vertices.  The final
    # assignment for a vertex v in that component is the XOR (across those k bits) of whichever
    # vectors have bit v=1.  This ensures for each v, it is 1 in at least one chosen vector => final
    # X_v != 0.  And the whole assignment is still in the nullspace => adjacency XOR = 0.

    # Step: we need adjacency presence in the component to decide if it's an "edgeful" component
    # or just an isolated vertex.  Let's quickly check that by counting the sum of degrees
    # restricted to the component.

    # Mark final answers in X array.  We'll do one component at a time.  For convenience,
    # we only need to handle the sub-component with edges (any vertex with deg >=1).
    # Isolated vertices can be assigned "1".

    deg_ = [len(adj[i]) for i in range(N)]
    # We only do the subgraph with edges if the component has at least one vertex with deg>0.

    # Function to build the final assignment for one component
    def assign_component(comp):
        # check if at least one vertex has deg>0 => if none => just set them =1
        cdeg = sum(deg_[v] for v in comp)
        if cdeg == 0:
            # all isolated, just set 1
            for v in comp:
                X[v] = 1
            return True

        # We have edges, find nullspace basis
        basis = find_nullspace_basis(comp)
        if basis is None or len(basis) == 0:
            # No nontrivial solution
            return False

        csize = len(comp)

        # For convenience, build a list of sets: which basis vectors have bit i set?
        # i.e. for each i in [0..csize-1], store list of basis_idx where basis[b] has bit i=1
        # If a particular i has none => no solution for that vertex => fail
        # We'll label "i" as the index of comp[i] in local ordering.
        coverage = [[] for _ in range(csize)]
        for b_idx, bmask in enumerate(basis):
            # check bits
            bm = bmask
            i_pos = 0
            while bm:
                lowb = (bm & -bm)
                pos = lowb.bit_length()-1
                coverage[pos].append(b_idx)
                bm ^= lowb

        # Now we do a simple set cover in the sense that we want each i covered by at least
        # one basis vector we choose. We'll pick them greedily. Each pick can cover multiple i's.
        # But a simpler approach (since csize <= 60) is just pick exactly one basis vector for
        # each uncovered i. That might pick duplicates though. We'll store them in a set.

        chosen_bvecs = set()
        covered_count = 0
        covered_flag = [False]*csize
        for i in range(csize):
            if deg_[comp[i]] == 0:
                # isolated in a bigger component? That can't happen if cdeg>0, unless
                # it's truly disconnected from the rest. Actually let's just set it =1.
                covered_flag[i] = True
            else:
                # We need to ensure i is covered
                # if coverage[i] is empty => fail
                if not coverage[i]:
                    return False
        # Now pick greedily
        while True:
            not_covered = [i for i in range(csize)
                           if (deg_[comp[i]] > 0) and (not covered_flag[i])]
            if not not_covered:
                break
            i0 = not_covered[0]
            b_candidates = coverage[i0]
            # pick any, e.g. the first
            bpick = b_candidates[0]
            chosen_bvecs.add(bpick)
            # that covers all i where basis[bpick] has bit i=1
            bmask = basis[bpick]
            bm = bmask
            while bm:
                lowb = (bm & -bm)
                pos = lowb.bit_length()-1
                covered_flag[pos] = True
                bm ^= lowb
            if len(chosen_bvecs) > 60:
                # we only have 60 bits
                return False

        # OK we have chosen up to 60 basis vectors so that every needed vertex i is covered.
        chosen_bvecs = list(chosen_bvecs)
        # We'll assign each chosen basis vector a distinct bit among up to 60 bits.
        # The final assignment for each local index i is the XOR of those bits for which
        # the basis vector has bit i set.  Then we store that in X[ comp[i] ].

        # We do this by simple construction: the j-th chosen vector uses bit j in the final.
        # So the final bitmask for i is the XOR over all j in chosen_bvecs where basis[j] has i-th bit set
        # i.e. if i-th bit is set in basis chosen_bvecs[k], we set bit k in the final integer for i.

        # But note that the final result for i is an integer up to 2^60-1.  We'll store it in X[ comp[i] ].
        # We'll do: for each k in [0..len(chosen_bvecs)-1], that corresponds to the bit (1<<k). 
        # If basis[ chosen_bvecs[k] ] has i-th bit => we add that bit to X.

        for i in range(csize):
            val = 0
            if deg_[comp[i]] == 0:
                # isolated => set 1
                X[comp[i]] = 1
                continue
            # else build from chosen basis combos
            for bitpos, bidx in enumerate(chosen_bvecs):
                bmask = basis[bidx]
                if ((bmask >> i) & 1) == 1:
                    val |= (1 << bitpos)
            # if val == 0 => fail => but theoretically it should not happen
            if val == 0:
                # might happen if i wasn't covered? => fail
                return False
            X[comp[i]] = val
        return True

    # Process each component
    for comp in comps:
        if not assign_component(comp):
            print("No")
            return

    # If we reach here, we have an assignment X that satisfies the conditions
    # We just need to verify none exceed 2^60-1. They won't exceed if we used at most 60 bits.
    # Also check that none are 0. Our code ensures any vertex with edges is nonzero, and
    # isolated vertices are assigned 1.  Finally, print result.
    # Double-check none is 0:
    for val in X:
        if val == 0:
            # Should never happen
            print("No")
            return
        if val >= (1 << 60):
            # Should not happen from our construction, but just in case
            print("No")
            return

    print("Yes")
    print(" ".join(str(val) for val in X))