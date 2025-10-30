import sys
from bisect import bisect_left, bisect_right

def solve():
    """
    This function solves the problem by calculating the number of safe squares.
    The grid is too large to simulate (N up to 10^9), so a combinatorial approach is needed.
    The number of pieces M is small (up to 1000), suggesting a solution polynomial in M.

    The core idea is to count safe squares row by row. A square is safe if it's not
    attacked by any existing piece. A piece at (r, c) attacks a square (i, j) if:
    i=r (same row), j=c (same column), i-j=r-c (same main diagonal), or i+j=r+c (same anti-diagonal).

    1. Identify Attacked "Lines":
       First, we find all unique rows, columns, main diagonals (identified by r-c), and
       anti-diagonals (identified by r+c) that are occupied by the M pieces.
       - R_set: set of occupied rows.
       - C_set: set of occupied columns.
       - D1_set: set of occupied main diagonals (r-c values).
       - D2_set: set of occupied anti-diagonals (r+c values).

    2. Process Rows in Strips:
       Any row `i` that contains an existing piece (i.e., `i` is in R_set) is fully attacked.
       Thus, there are zero safe squares in these rows.
       We only need to consider rows that do not contain any pieces. These "safe" rows can be
       grouped into contiguous blocks, or "strips", between the occupied rows.
       We can iterate through these strips. For a strip of rows from `i_min` to `i_max`, all
       rows `i` in this range are not in R_set.

    3. Counting Safe Squares in a Strip:
       For each row `i` in a safe strip `[i_min, i_max]`, we need to count the number of
       columns `j` such that the square `(i, j)` is safe. A square `(i, j)` is safe if:
       - Column `j` is not in C_set.
       - Main diagonal `i-j` is not in D1_set.
       - Anti-diagonal `i+j` is not in D2_set.

       Let Forbidden_j(i) be the set of columns `j` that are attacked for a given row `i`.
       The number of safe squares in row `i` is `N - |Forbidden_j(i)|`.
       The total safe squares in the strip is `Sum_{i=i_min to i_max} (N - |Forbidden_j(i)|)`.

    4. Principle of Inclusion-Exclusion (PIE):
       `|Forbidden_j(i)|` is the size of the union of three sets of forbidden columns:
       - `U_C = {j | j is in C_set}`
       - `U_D1(i) = {j | i-j is in D1_set}`
       - `U_D2(i) = {j | i+j is in D2_set}`
       We use PIE to calculate `|U_C U U_D1(i) U U_D2(i)|`.
       The total number of attacked squares in the strip is `Sum_{i} |U_C U U_D1(i) U U_D2(i)|`.
       By linearity of summation, this becomes `(Sum |U_C|) + (Sum |U_D1(i)|) + ...`

    5. Efficiently Calculating PIE Terms:
       Each term of the PIE expansion summed over `i` in the strip `[i_min, i_max]` can be
       calculated efficiently.
       - Simple terms like `Sum |U_C|` are straightforward.
       - Terms like `Sum |U_D1(i)|` can be calculated by changing the order of summation.
       - Intersection terms like `Sum |U_C intersect U_D1(i)|` correspond to counting
         "events" where `i = c + d1` for `c` in C_set, `d1` in D1_set. We can pre-calculate
         all such event points `i`, sort them, and use binary search (`bisect`) to count
         how many fall within the current strip `[i_min, i_max]`.

    The overall complexity is dominated by pre-calculating and sorting the event points,
    which is O(M^2 * log M), and then iterating through the O(M) strips, which takes
    O(M) time per strip, for a total of O(M^2). The final complexity is O(M^2 * log M),
    which is efficient enough for M <= 1000.
    """
    try:
        input = sys.stdin.readline
        N, M = map(int, input().split())
        if M == 0:
            print(N * N)
            return
            
        pieces = [tuple(map(int, input().split())) for _ in range(M)]
    except (IOError, ValueError):
        # Handle empty input or format errors gracefully
        # This part is for robustness, e.g., in local testing
        N, M = 8, 6
        pieces = [(1,4),(2,1),(3,8),(4,5),(5,2),(8,3)]
        if M == 0:
           print(N * N)
           return
           
    R_set = set()
    C_set = set()
    D1_set = set()  # r - c
    D2_set = set()  # r + c

    for r, c in pieces:
        R_set.add(r)
        C_set.add(c)
        D1_set.add(r - c)
        D2_set.add(r + c)

    C_list = sorted(list(C_set))
    D1_list = sorted(list(D1_set))
    D2_list = sorted(list(D2_set))

    # Pre-calculate event points for intersection terms
    events_C_D1 = sorted([c + d1 for c in C_list for d1 in D1_list])
    events_C_D2 = sorted([d2 - c for c in C_list for d2 in D2_list])
    
    events_D1_D2 = []
    for d1 in D1_list:
        for d2 in D2_list:
            if (d1 + d2) % 2 == 0:
                j = (d2 - d1) // 2
                if 1 <= j <= N:
                    i = (d1 + d2) // 2
                    events_D1_D2.append(i)
    events_D1_D2.sort()

    events_C_D1_D2 = []
    d2_lookup = set(D2_list)
    for c in C_list:
        for d1 in D1_list:
            d2_target = 2 * c + d1
            if d2_target in d2_lookup:
                i = c + d1
                events_C_D1_D2.append(i)
    events_C_D1_D2.sort()

    total_safe_count = 0
    
    r_coords = sorted(list(R_set))
    boundaries = [0] + r_coords + [N + 1]

    for k in range(len(boundaries) - 1):
        r_start = boundaries[k]
        r_end = boundaries[k + 1]
        
        i_min = r_start + 1
        i_max = r_end - 1

        if i_min > i_max:
            continue

        num_rows = i_max - i_min + 1
        
        # Calculate attacked squares in this strip using PIE
        
        # T1 = Sum |U_C'|
        T1 = num_rows * len(C_list)
        
        # T2 = Sum |U_D1(i)'|
        T2 = 0
        for d in D1_list:
            start = max(i_min, d + 1)
            end = min(i_max, d + N)
            if start <= end:
                T2 += (end - start + 1)

        # T3 = Sum |U_D2(i)'|
        T3 = 0
        for d in D2_list:
            start = max(i_min, d - N)
            end = min(i_max, d - 1)
            if start <= end:
                T3 += (end - start + 1)
        
        # T4 = Sum |U_C' n U_D1(i)'|
        T4 = bisect_right(events_C_D1, i_max) - bisect_left(events_C_D1, i_min)
        
        # T5 = Sum |U_C' n U_D2(i)'|
        T5 = bisect_right(events_C_D2, i_max) - bisect_left(events_C_D2, i_min)

        # T6 = Sum |U_D1(i)' n U_D2(i)'|
        T6 = bisect_right(events_D1_D2, i_max) - bisect_left(events_D1_D2, i_min)
        
        # T7 = Sum |U_C' n U_D1(i)' n U_D2(i)'|
        T7 = bisect_right(events_C_D1_D2, i_max) - bisect_left(events_C_D1_D2, i_min)

        total_attacked_in_strip = T1 + T2 + T3 - (T4 + T5 + T6) + T7
        
        strip_total_squares = num_rows * N
        strip_safe_squares = strip_total_squares - total_attacked_in_strip
        total_safe_count += strip_safe_squares

    print(total_safe_count)

solve()