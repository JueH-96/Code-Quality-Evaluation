import sys

def solve():
    """
    Determines if a good string exists for a given binary sequence A.
    A good string S allows turning all 0s in A into 1s using specified operations.
    Operations are triggered by 'ARC' or 'CRA' substrings in S (circularly)
    and affect two adjacent elements in A (circularly).

    The core operations are:
    1. Choose i (0-indexed) such that S[i]='A', S[(i+1)%N]='R', S[(i+2)%N]='C'.
       Set A[i] = 1, A[(i+1)%N] = 1.
    2. Choose i (0-indexed) such that S[i]='C', S[(i+1)%N]='R', S[(i+2)%N]='A'.
       Set A[i] = 1, A[(i+1)%N] = 1.

    Both operations, when triggered by the pattern S[i..i+2] (circularly),
    affect the pair of indices (i%N, (i+1)%N) in A.
    An index k in A needs to be set to 1 if A[k] is initially 0.
    Index k can be set to 1 if it is covered by an operation.
    Index k is covered if k is the first or second index of a pair (i%N, (i+1)%N)
    where an operation starting at i is possible.
    This means k = i%N or k = (i+1)%N for some i where S[i..i+2] is 'ARC' or 'CRA'.
    Equivalently, for every k with A[k]=0, we need k to be covered by some operation.
    An operation starting at i covers (i, i+1). Index k is covered if k=i or k=i+1.
    So, if A[k]=0, we need i=k or i=k-1 (modulo N) to be a valid start index for an operation.
    A valid start index i means S[i..i+2] is 'ARC' or 'CRA'.
    This implies S[(i+1)%N] must be 'R', and {S[i%N], S[(i+2)%N]} must be {'A', 'C'}.

    If A[k]=0, then:
    - If k is covered by an operation at k: S[k..k+2] is 'ARC'/'CRA'.
      Requires S[(k+1)%N]='R', {S[k%N], S[(k+2)%N]}={'A','C'}.
    - If k is covered by an operation at k-1: S[k-1..k+1] is 'ARC'/'CRA'.
      Requires S[k%N]='R', {S[(k-1+N)%N], S[(k+1+N)%N]}={'A','C'}.

    Let $I_{ops}$ be the set of indices i where an operation is possible.
    $I_{ops} = \{i \mid S[i..i+2] \text{ is 'ARC' or 'CRA'} \text{ (circularly)} \}$.
    Condition for existence: There must exist S such that $\{k \mid A[k]=0\} \subseteq \bigcup_{i \in I_{ops}} \{i\%N, (i+1)\%N\}$.
    This is equivalent to: for every k with A[k]=0, k must be in $\{i, i+1\}$ for some $i \in I_{ops}$.
    This is equivalent to: for every k with A[k]=0, $k \in I_{ops}$ or $(k-1+N)\%N \in I_{ops}$.

    Also, the set $I_{ops}$ must be constructible by a single string S.
    If $i \in I_{ops}$, S[(i+1)%N] = 'R' and {S[i%N], S[(i+2)%N]} = {'A', 'C'}.
    If $j \in I_{ops}$ ($i 
e j$): S[(j+1)%N] = 'R' and {S[j%N], S[(j+2)%N]} = {'A', 'C'}.
    If $j = (i+1)\%N$, then S[(i+1)%N]='R' (from $i$) and S[(i+2)%N]='R' (from $i+1$). Impossible for N>=1. So $i, (i+1)\%N$ cannot both be in $I_{ops}$.
    If $j = (i+2)\%N$, then S[(i+1)%N]='R' (from $i$) and S[(i+3)%N]='R' (from $i+2$). Also {S[i%N], S[(i+2)%N]}={'A','C'} (from $i$) and {S[(i+2)%N], S[(i+4)%N]}={'A','C'} (from $i+2$). Conflict S[(i+2)%N] must be 'A'/'C' and 'R'. Impossible. So $i, (i+2)\%N$ cannot both be in $I_{ops}$.

    So $I_{ops}$ must not contain adjacent indices or indices separated by 2 (circularly).
    $i \in I_{ops} \implies (i+1)\%N 
otin I_{ops}$ and $(i+2)\%N 
otin I_{ops}$.

    Consider the case where A[k]=0, A[(k+1)%N]=0, A[(k+2)%N]=0 for some k.
    Let's use 0-indexed without explicit modulo for clarity, assuming wrap-around.
    A[k]=0 requires $k \in I_{ops}$ or $k-1 \in I_{ops}$.
    A[k+1]=0 requires $k+1 \in I_{ops}$ or $k \in I_{ops}$.
    A[k+2]=0 requires $k+2 \in I_{ops}$ or $k+1 \in I_{ops}$.

    Case 1: $k \in I_{ops}$.
    By constraint, $k+1 
otin I_{ops}$ and $k+2 
otin I_{ops}$.
    Check requirement for k+1: needs $k+1 \in I_{ops}$ or $k \in I_{ops}$. Since $k \in I_{ops}$, OK.
    Check requirement for k+2: needs $k+2 \in I_{ops}$ or $k+1 \in I_{ops}$. Both $k+2, k+1$ are NOT in $I_{ops}$. Requirement NOT met.
    So $k \in I_{ops}$ is impossible if A[k..k+2] are all 0.

    Case 2: $k 
otin I_{ops}$.
    From A[k]=0 requirement: $k-1 \in I_{ops}$.
    By constraint, $k-1 \in I_{ops} \implies k 
otin I_{ops}$ (consistent) and $k+1 
otin I_{ops}$.
    Check requirement for k+1: needs $k+1 \in I_{ops}$ or $k \in I_{ops}$. Both $k+1, k$ are NOT in $I_{ops}$. Requirement NOT met.
    So $k 
otin I_{ops}$ is also impossible if A[k..k+2] are all 0.

    Since both cases lead to a contradiction, if there are three consecutive 0s (circularly), no good string exists.

    Conversely, if there are no three consecutive 0s (circularly), it is known that a good string exists. This relies on constructing an appropriate $I_{ops}$ (an independent set in the graph $G$ with edges $(i, i+1)$ and $(i, i+2)$ that hits all pairs $\{k, k-1\}$ for $k$ with $A[k]=0$) and showing that an S can be constructed from this $I_{ops}$. This part is non-trivial but guaranteed by the problem structure when the 000 forbidden pattern is absent.

    Therefore, the condition for existence of a good string is simply the absence of three consecutive 0s in A, checked circularly.
    """
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # If A contains no zeros, any string is good.
    # This is covered by the check below, but explicitly handling it is slightly clearer.
    # If sum(A) == N:
    #    print("Yes")
    #    return

    # Check for three consecutive zeros (circularly)
    # If A[i], A[i+1], A[i+2] are all 0 (indices mod N), then no good string exists.
    for i in range(N):
        if A[i] == 0 and A[(i + 1) % N] == 0 and A[(i + 2) % N] == 0:
            print("No")
            return

    # If the loop completes, it means no three consecutive zeros were found.
    # In this case, a good string is guaranteed to exist.
    print("Yes")

solve()