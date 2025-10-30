# YOUR CODE HERE
import sys

def main():
    """
    Main function to read input and solve all test cases.
    """
    try:
        T_str = sys.stdin.readline()
        if not T_str:
            return
        T = int(T_str)
        for _ in range(T):
            solve()
    except (IOError, ValueError):
        # Handles empty input or malformed integers
        return

def solve():
    """
    Solves a single test case.
    """
    try:
        N_str = sys.stdin.readline()
        if not N_str: return
        N = int(N_str)
        A = sys.stdin.readline().strip()
        B = sys.stdin.readline().strip()
    except (IOError, ValueError):
        return

    pos_A = [i for i, char in enumerate(A) if char == '1']
    pos_B = [i for i, char in enumerate(B) if char == '1']

    m = len(pos_A)
    l = len(pos_B)

    # Each target position in B must be occupied by at least one piece.
    # The m pieces from A can occupy at most m distinct positions.
    # Therefore, if there are more target positions than available pieces, it's impossible.
    if m < l:
        print(-1)
        return

    # check(k) determines if it's possible to achieve the target configuration
    # with a maximum cost of k. The cost is defined as max(|displacement|).
    def check(k):
        # We greedily assign pieces from A to target positions in B.
        # last_a_idx is the index in pos_A of the last piece assigned to the previous group.
        last_a_idx = -1
        
        # Iterate through each target group corresponding to positions in pos_B.
        for j in range(l):
            b_j = pos_B[j]
            
            # The current group must start with the piece after the last assigned one.
            start_a_idx = last_a_idx + 1
            if start_a_idx >= m:
                return False  # Not enough pieces left in A for the remaining groups.
            
            a_start = pos_A[start_a_idx]
            
            # A key property of the operations is that the displacement function
            # d_i = q_i - a_i is non-increasing, i.e., d_i >= d_{i+1}.
            # This must hold at the boundary between groups.
            if j > 0:
                b_prev = pos_B[j-1]
                a_last_of_prev_group = pos_A[last_a_idx]
                if b_prev - a_last_of_prev_group < b_j - a_start:
                    return False
            
            # The displacement of any piece must be within [-k, k].
            # For the first piece of the current group, displacement is b_j - a_start.
            # b_j - a_start <= k  => a_start >= b_j - k
            if a_start < b_j - k:
                return False

            # Find the smallest possible end index for this group.
            # Choosing the smallest valid end index is a greedy strategy that maximizes
            # flexibility for subsequent groups.
            end_a_idx = -1
            if j == l - 1:
                # The last group must take all remaining pieces.
                # Check if the last piece's displacement is valid.
                if pos_A[m-1] <= b_j + k:
                    end_a_idx = m - 1
                else:
                    return False
            else:
                b_next = pos_B[j+1]
                gap_needed = b_next - b_j
                
                # We need to find the first piece `i` that can end the current group.
                # Total work for these linear scans across all j is O(m).
                found = False
                for i in range(start_a_idx, m - 1):
                    # Displacement of piece i must be valid: pos_A[i] <= b_j + k
                    if pos_A[i] > b_j + k:
                        break  # All subsequent pieces will also fail this condition.
                    
                    # Boundary condition for the next group.
                    if pos_A[i+1] - pos_A[i] >= gap_needed:
                        end_a_idx = i
                        found = True
                        break
                if not found:
                    return False

            last_a_idx = end_a_idx

        # If we successfully assigned all pieces and covered all groups.
        return last_a_idx == m - 1

    # Binary search for the minimum possible cost k.
    low = 0
    high = N 
    ans = -1

    while low <= high:
        mid = (low + high) // 2
        if check(mid):
            ans = mid
            high = mid - 1
        else:
            low = mid + 1
    
    print(ans)

if __name__ == "__main__":
    main()