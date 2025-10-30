# YOUR CODE HERE
import sys
from collections import deque

def solve():
    # Read input
    N, M = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()

    # possible_mask[i] is a bitmask where the k-th bit is set if position i
    # can potentially be matched by T[k] as determined by the LAST operation
    # covering position i.
    # Initially, this includes all k such that S[i] == T[k] and an operation
    # starting at j = i-k is within valid bounds [0, N-M].
    possible_mask = [0] * N
    q = deque()

    # Initialize possibilities based on character match and valid start index
    # For position i, it can be matched by T[k] if an operation starting at j=i-k
    # is valid (0 <= j <= N-M) and S[i] == T[k].
    for i in range(N):
        current_mask = 0
        for k in range(M):
            j = i - k
            # Check if operation starting at j is valid (covers range [j, j+M-1])
            # 0 <= j <= N - M
            if 0 <= j and j <= N - M:
                # Check if character matches T[k]
                if S[i] == T[k]:
                    current_mask |= (1 << k) # Set k-th bit

        if current_mask == 0:
            # If no possible T-offset can match S[i] from a valid operation, impossible.
            print("No")
            return

        possible_mask[i] = current_mask
        # Add all positions to the queue initially as their initial state might constrain others.
        q.append(i)

    # Propagate constraints
    # The core constraint: If position i is LAST covered by an operation starting at j=i-l
    # (where l is a possible offset for i, meaning (possible_mask[i] >> l) & 1 is true),
    # then for ANY position p covered by this same operation ([j, j+M-1]),
    # the LAST operation covering p must start at j' >= j.
    # Let p = j + offset, where 0 <= offset < M.
    # The last op on p starts at j' = p - k' where k' is the offset in T for position p.
    # We need j' >= j => (p - k') >= j => (j + offset - k') >= j => offset >= k'.
    # So, for any position p covered by the operation starting at j=i-l, any currently
    # possible offset k' for p (where (possible_mask[p] >> k') & 1 is true) must satisfy k' <= offset.

    # We iterate through positions i whose possible_mask has been updated.
    # For each possible offset l at position i, we consider the operation starting at j=i-l.
    # This potential operation constraints other positions p it covers.
    while q:
        i = q.popleft()
        current_mask_i = possible_mask[i] # Current mask of possible T-offsets for position i

        # Iterate through each possible offset l for position i (each set bit in current_mask_i)
        # Using a simple loop 0 to M-1 as M is small
        for l in range(M):
            # Check if l-th bit is set in the current mask for i
            if (current_mask_i >> l) & 1:
                # If l is a possible offset for i, it means position i could be T[l]
                # from a LAST operation starting at j = i - l.
                j = i - l # Potential start index of the last operation covering i

                # This operation covers indices [j, j+M-1]
                # Propagate constraint to all positions p covered by this potential operation
                for offset in range(M):
                    p = j + offset # Position covered by this operation

                    # Check if p is a valid index within the string S
                    if 0 <= p < N:
                        # Constraint: for position p, any possible T-offset k' must satisfy k' <= offset
                        required_max_k_prime = offset

                        # Create a mask allowing bits from 0 up to required_max_k_prime (inclusive)
                        # (1 << (n+1)) - 1 gives a mask with bits 0 to n set.
                        # Example: required_max_k_prime=0 => mask=(1<<1)-1=1 (001), allows k'=0
                        # Example: required_max_k_prime=1 => mask=(1<<2)-1=3 (011), allows k'=0, 1
                        # Example: required_max_k_prime=2 => mask=(1<<3)-1=7 (111), allows k'=0, 1, 2
                        allowed_mask = (1 << (required_max_k_prime + 1)) - 1

                        # Apply the constraint: intersect the current mask for p with the allowed mask
                        new_mask_p = possible_mask[p] & allowed_mask

                        # Check for contradiction: if the mask becomes 0, it's impossible
                        if new_mask_p == 0:
                            print("No")
                            return

                        # If the mask for p has changed (new constraints were applied)
                        if new_mask_p != possible_mask[p]:
                            # Update the mask for p
                            possible_mask[p] = new_mask_p
                            # Add p to the queue so its new state can propagate further constraints
                            q.append(p)

    # If the loop completes without any mask becoming 0, it means there exists at least one
    # valid assignment of last-covering operations consistent with S.
    print("Yes")

solve()