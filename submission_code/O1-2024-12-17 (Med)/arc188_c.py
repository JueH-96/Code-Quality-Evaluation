def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    N, M = map(int, input_data[:2])
    # Testimonies: A_i, B_i, C_i (0-based indexing later)
    ABC = input_data[2:]
    
    # ----------------------------------------------------------------
    # Explanation of the approach:
    #
    # We have N villagers (1..N). Each testimony is (A, B, C):
    #   - C=0 means "A says B is honest"
    #   - C=1 means "A says B is a liar"
    #
    # We denote:
    #   X(i) in {0,1}, meaning villager i is honest (0) or a liar (1).
    #   Y(i) in {0,1}, meaning villager i is not confused (0) or confused (1).
    #
    # The rule for a testimony (A,B,C) is:
    #   X(A) XOR Y(A) = P(A), where:
    #       P(A) = 0 means "A"'s statement is 'as is'
    #       P(A) = 1 means "A"'s statement is reversed
    #   Then the testimony says: X(B) = C XOR P(A).
    #   Rearranging: X(A) XOR X(B) = C XOR Y(A).
    #
    # From a single A with testimonies about B1..Bk (with labels C1..Ck),
    # we get that X(Bi) XOR C_i must be constant for all i, because
    #   X(A) XOR X(Bi) = C_i XOR Y(A)  =>  X(Bi) XOR C_i = X(A) XOR Y(A),  a common value.
    #
    # So let's define L(A) = X(A) XOR Y(A).  Then for each testimony (A,B,C):
    #   we have X(B) = L(A) XOR C.
    #
    # If a villager B is testified-about by multiple A_i's, say (A1,B,C1) and (A2,B,C2),
    # then we must have L(A1) XOR C1 = L(A2) XOR C2, hence
    #   L(A1) XOR L(A2) = C1 XOR C2.
    #
    # Therefore, we only need to assign an unknown bit L(A) for each villager A who TESTIFIES
    # (i.e. appears as the "A" in some testimony).  We gather all testimonies about the same B
    # to create edges among the A_i's who mention that same B.  An edge between A1 and A2
    # has "weight" = (C1 XOR C2).  This means L(A1) XOR L(A2) must equal that weight.
    #
    # We then do a graph consistency check on these L(A) variables:
    #   - Build a graph whose nodes are exactly the villagers who appear as a testifier
    #   - For each B, gather the testifiers about B, pick one as an anchor, and connect
    #     all others to it with the appropriate XOR-weight.  This ensures we do not explode
    #     into cliques (we only add k-1 edges if k testifiers mention B).
    #   - Then we run a DFS or BFS to see if there is a consistent assignment of L(A)
    #     in each connected component.  If we find a contradiction, output -1.
    #
    # Once L(A) is fixed for all testifiers A, we can assign:
    #   - For each B (whether or not B itself is a testifier), if it is mentioned by at least
    #     one testifier A, we define X(B) = L(A) XOR C for that testimony.  If multiple
    #     testifiers mention B, they all must agree, otherwise contradiction => -1.
    #   - For any villager v never mentioned (no testimonies about v), we can set X(v)=0 freely.
    #
    # Then for each testifier A:
    #   - If A is also mentioned by some other testifier A', we have X(A) = L(A') XOR C'
    #     (from the testimony (A',A,C')) and that must be consistent if multiple testifiers
    #     mention A.
    #   - If no one ever testifies about A, we set X(A)=0 arbitrarily.
    #
    # Finally, we get Y(A) = X(A) XOR L(A) for each A who testifies.
    # For a villager i who never testifies, Y(i) can be 0 (since it's unconstrained).
    #
    # If all is consistent, we print out the string of length N for Y(1..N).  Otherwise -1.
    #
    # Time complexity is O(M + N) because we construct adjacency with at most M edges 
    # and then do BFS/Union-Find to assign L-values.
    # ----------------------------------------------------------------
    
    # Read testimonies into arrays (A_i, B_i, C_i) as 0-based
    # Also track which villagers appear as "A" (testifiers).
    testifier_set = set()
    idx = 0
    testimonies = []
    for _ in range(M):
        A_ = int(ABC[idx]); B_ = int(ABC[idx+1]); C_ = int(ABC[idx+2])
        idx += 3
        A_ -= 1  # make 0-based
        B_ -= 1  # make 0-based
        # C_ is already 0 or 1
        testimonies.append((A_, B_, C_))
        testifier_set.add(A_)
    
    # Build a mapping "testifier_id[v]" -> an integer index in [0..num_testifiers-1]
    # or -1 if v is not a testifier.
    # We'll store those in an array of length N, initialized to -1.
    testifier_id = [-1]*N
    testifiers = sorted(testifier_set)
    for i, t in enumerate(testifiers):
        testifier_id[t] = i
    
    numT = len(testifiers)  # number of distinct testifiers
    
    # For each villager B, gather the list of (A_i, C_i) where A_i testifies about B.
    # We'll store this in "mentioned_by[B] = list of (A_i, C_i)".
    mentioned_by = [[] for _ in range(N)]
    for A_, B_, C_ in testimonies:
        mentioned_by[B_].append((A_, C_))
    
    # We will build an adjacency list among the testifiers (0..numT-1).
    # Each edge is (u, v, w) meaning L(u) XOR L(v) = w.
    # We'll store as adjacency[u] = list of (v, w).
    adjacency = [[] for _ in range(numT)]
    
    # Now, for each B, if it is mentioned by k>1 distinct testifiers, we pick an anchor
    # (A1, C1) and for each other (A2, C2) we add an edge testifier_id[A1] <-> testifier_id[A2]
    # with weight = C1 XOR C2.  This enforces L(A1) XOR L(A2) = (C1 XOR C2).
    # We do this only if both A1 and A2 are indeed testifiers (their IDs != -1).
    # Note: If k=1, there's no constraint among testifiers for that B.  If k=0, B is not
    # mentioned at all.
    for B in range(N):
        arr = mentioned_by[B]
        k = len(arr)
        if k <= 1:
            continue
        # pick anchor arr[0]
        A1, C1 = arr[0]
        id1 = testifier_id[A1]
        # for j in [1..k-1], connect anchor with arr[j]
        for j in range(1, k):
            A2, C2 = arr[j]
            id2 = testifier_id[A2]
            if id1 == -1 or id2 == -1:
                # If one is not a testifier, no L() constraints among them.
                # (Though we do not ignore the possibility that B is also a testifier,
                #  but that is handled separately if B is itself "A" in some testimony.)
                continue
            w = C1 ^ C2
            adjacency[id1].append((id2, w))
            adjacency[id2].append((id1, w))
    
    # We now see if we can assign L(A) for each A in [0..numT-1] consistently.
    # We'll do a DFS/BFS over the graph.  compVal[a] = -1 if unassigned; else in {0,1}.
    compVal = [-1]*numT
    
    from collections import deque
    
    for start in range(numT):
        if compVal[start] == -1:
            # assign compVal[start] = 0 arbitrarily and propagate
            compVal[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for (v, w) in adjacency[u]:
                    desired = compVal[u] ^ w
                    if compVal[v] == -1:
                        compVal[v] = desired
                        queue.append(v)
                    else:
                        if compVal[v] != desired:
                            # Contradiction
                            print(-1)
                            return
    
    # If we got here, we have a consistent assignment for L(A) for all testifier A.
    # Next, we assign X(i) for i in [0..N-1].
    # Initialize X(i) = -1 => unassigned.
    X = [-1]*N
    
    # For each B, if it is mentioned by at least one testifier, pick an anchor mention
    # to define X(B).  Then check consistency with all other mentions.
    for B in range(N):
        arr = mentioned_by[B]
        if not arr:
            # B not mentioned by anyone => X(B) can be 0
            X[B] = 0
            continue
        # pick the first mention as anchor
        A1, C1 = arr[0]
        if testifier_id[A1] != -1:
            L_A1 = compVal[testifier_id[A1]]
            val = L_A1 ^ C1
        else:
            # if A1 is not in W, L(A1) doesn't exist,
            # but that also means there's no constraint on L(A1).
            # We can set L(A1) = 0 effectively, so X(B) = 0 ^ C1 = C1
            # but actually any choice is fine. Let's pick 0 for L(A1).
            val = 0 ^ C1
        
        # Now X(B) should be val, check with others
        for j in range(1, len(arr)):
            A2, C2 = arr[j]
            if testifier_id[A2] != -1:
                L_A2 = compVal[testifier_id[A2]]
                check_val = L_A2 ^ C2
            else:
                # similarly pick 0 for L(A2)
                check_val = 0 ^ C2
            if check_val != val:
                # Contradiction
                print(-1)
                return
        
        # Now we can assign X(B) = val
        X[B] = val
    
    # Now, for each testifier A, if A was testified about by others, that also defines X(A).
    # So let's check consistency: for testimonies (A', A, C') that mention A as B,
    # we must have X(A) = compVal[testifier_id[A']] ^ C' if A' is a testifier,
    # otherwise 0 ^ C' if A' is not a testifier.  If there's a mismatch => contradiction.
    # If X(A) is still unassigned, we set it to 0 by default.
    
    # We'll build for each "villager i" the set of (A', C') where i = B in that testimony.
    # Actually, we already have that in mentioned_by[i].  Let's just check i in testifier_set.
    # Because only then do we care about X(i) from "i is a testifier" perspective.
    
    for i in testifier_set:
        # i is a testifier. If X(i) < 0 we might set it to 0, but let's check if i is mentioned:
        if X[i] == -1:
            # Not mentioned by anyone => set X(i)=0
            X[i] = 0
        else:
            # If i is mentioned by k>0 testifiers, we already assigned X(i).
            # Double check consistency:
            arr = mentioned_by[i]
            assigned_val = X[i]
            for (A2, C2) in arr:
                if testifier_id[A2] != -1:
                    check_val = compVal[testifier_id[A2]] ^ C2
                else:
                    check_val = 0 ^ C2
                if check_val != assigned_val:
                    print(-1)
                    return
    
    # Any villager i not in testifier_set but X(i) == -1 means i was not mentioned => set X(i)=0
    for i in range(N):
        if X[i] == -1:
            X[i] = 0
    
    # Now we have X(i) for all i and compVal[] = L(A) for all A in W.
    # Finally, Y(i) = X(i) XOR L(i) if i is a testifier, else Y(i)=0.
    Y = [0]*N
    for a_idx, a_vill in enumerate(testifiers):
        Y[a_vill] = X[a_vill] ^ compVal[a_idx]
    
    # Output the result as a string of length N of '0'/'1'.
    print("".join(str(y) for y in Y))