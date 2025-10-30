import sys

def solve():
    """
    This function solves the problem of finding the floor((S+1)/2)-th good integer sequence.
    """
    try:
        # Fast I/O
        input = sys.stdin.readline
        
        N, K = map(int, input().split())
        
        L = N * K
        
        counts = [0] * (N + 1)
        for i in range(1, N + 1):
            counts[i] = K
            
        ans = []
        
        is_less = False
        
        # This function determines if a multiset (represented by counts)
        # can be arranged into a sequence P such that P is lexicographically
        # less than or equal to T(P), where T is the transformation x -> N+1-x.
        def is_valid(current_counts):
            v_min = -1
            for i in range(1, N + 1):
                if current_counts[i] > 0:
                    v_min = i
                    break
            
            if v_min == -1: # Empty multiset
                return True

            # If the smallest available number v_min satisfies v_min < (N+1)/2,
            # we can start a sequence with it, ensuring P < T(P).
            if 2 * v_min < N + 1:
                return True
            
            # If v_min > (N+1)/2, any sequence will start with an element > (N+1)/2,
            # forcing P > T(P).
            if 2 * v_min > N + 1:
                return False
            
            # If v_min = (N+1)/2, we must start with v_min. For the sequence to remain
            # valid (P' <= T(P')), the rest of the multiset must also be valid.
            # This holds recursively only if there are no elements larger than v_min.
            else: # 2 * v_min == N + 1
                v_max = -1
                for i in range(N, 0, -1):
                    if current_counts[i] > 0:
                        v_max = i
                        break
                
                return v_min == v_max

        for _ in range(L):
            for v in range(N, 0, -1):
                if counts[v] == 0:
                    continue
                
                if is_less:
                    # The condition P < T(P) is already guaranteed.
                    # We greedily pick the largest available number to make the sequence lexicographically maximal.
                    ans.append(v)
                    counts[v] -= 1
                    break
                else:
                    # We are on the boundary, trying to find the largest v for P <= T(P).
                    
                    # If 2*v > N+1, picking v would make P > T(P). Invalid choice.
                    if 2 * v > N + 1:
                        continue
                    
                    counts[v] -= 1
                    
                    # If 2*v < N+1, picking v guarantees P < T(P). This is a valid choice.
                    # Since we iterate v downwards, this is the best such choice.
                    if 2 * v < N + 1:
                        ans.append(v)
                        is_less = True
                        break
                    
                    # If 2*v == N+1, we must check if the remaining multiset is valid.
                    if is_valid(counts):
                        ans.append(v)
                        # is_less remains False, we are still on the P=T(P) boundary path.
                        break
                    else:
                        # This choice of v does not lead to a valid completion. Backtrack.
                        counts[v] += 1
        
        print(*ans)

    except (IOError, IndexError) as e:
        # Handle potential errors if input is not as expected, though not required by problem spec
        pass

solve()