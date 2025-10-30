from collections import Counter
import sys

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))

    # 1. Check non-negative for non -1s
    for x in A:
        if x != -1 and x < 0:
            print("No")
            return
    for x in B:
        if x != -1 and x < 0:
            print("No")
            return

    # 2. Collect non -1s
    A_nz_list = [x for x in A if x != -1]
    B_nz_list = [x for x in B if x != -1]

    A_nz = Counter(A_nz_list)
    B_nz = Counter(B_nz_list)

    k = N - len(A_nz_list) # count of -1 in A
    l = N - len(B_nz_list) # count of -1 in B

    # 3. Find S
    S = None
    for i in range(N):
        if A[i] != -1 and B[i] != -1:
            current_S = A[i] + B[i]
            if S is None:
                S = current_S
            elif S != current_S:
                print("No")
                return

    if S is None: # S not fixed by Type 1 pairs
        if len(A_nz_list) != len(B_nz_list):
            print("No")
            return
        
        N_A = len(A_nz_list)
        if N_A > 0:
            A_nz_list.sort()
            B_nz_list.sort()
            
            S_candidate = A_nz_list[0] + B_nz_list[N_A - 1]
            for i in range(1, N_A):
                if A_nz_list[i] + B_nz_list[N_A - 1 - i] != S_candidate:
                    print("No")
                    return
            S = S_candidate
        else: # All A and B are -1. N_A=0, N_B=0. k=N, l=N. A_nz, B_nz are empty.
             # This case is always possible. Choose A_i=0, B_i=0. S=0.
             print("Yes")
             return

    # S is fixed now (either by Type 1 or Type 3 with N_1=0)
    # 4. Check S non-negativity and bounds from fixed values
    # S can be 0. If S=0, max(0, A_nz, B_nz) <= 0.
    
    max_val_nz = 0
    if A_nz_list:
        max_val_nz = max(max_val_nz, max(A_nz_list))
    if B_nz_list:
        max_val_nz = max(max_val_nz, max(B_nz_list))

    if S < max_val_nz:
        print("No")
        return
        
    # 5. Compute required A values from fixed B values
    R_Bnz = Counter()
    for b in B_nz_list:
        required_A = S - b
        # Check S-b >= 0 implicitly covered by S >= max_val_nz if B_nz_list is not empty
        # If B_nz_list is empty, R_Bnz is empty, check passes.
        R_Bnz[required_A] += 1

    # 6. Compare multisets using Counter arithmetic
    A_rem = A_nz - R_Bnz # Elements in A_nz not fully matched by R_Bnz
    R_rem = R_Bnz - A_nz # Elements in R_Bnz not fully matched by A_nz

    # 7. Check remaining counts vs free slots
    sum_A_rem = sum(A_rem.values())
    sum_R_rem = sum(R_rem.values())

    if sum_A_rem > l or sum_R_rem > k:
        print("No")
        return

    # 8. Check if remaining deficit can be filled by same number of free variables
    # The number of *additional* free variables required (beyond covering the unmatched fixed ones)
    # must be the same for both A and R_Bnz.
    m = l - sum_A_rem # Number of remaining free B slots after covering A_rem
    n = k - sum_R_rem # Number of remaining free A slots after covering R_rem
    if m != n:
         print("No")
         return
    # m must be >= 0, which is guaranteed by sum_A_rem <= l and sum_R_rem <= k

    # 10. Check element bounds for remaining elements
    # Elements in A_rem must be <= S.
    for val in A_rem:
        if A_rem[val] > 0 and val > S:
            print("No")
            return
    # Elements in R_rem must be >= 0. (Checked in step 5 when creating R_Bnz)
    # If B_nz_list was not empty, S >= max_val_nz >= any b, so S-b >= 0.
    # If B_nz_list was empty, R_Bnz is empty, R_rem is empty, check passes.

    print("Yes")

solve()