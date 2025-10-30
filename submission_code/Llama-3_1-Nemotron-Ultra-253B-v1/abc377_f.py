<think>
Okay, let's see. I need to solve this problem where I have to find the number of empty squares on a grid where placing a new piece won't be captured by any existing pieces. The grid is N x N, which can be really big, up to 1e9. And there are M existing pieces, up to 1e3. So the key is to process these M pieces efficiently without iterating through all possible squares, which is impossible for large N.

First, let's understand the capture conditions. A piece at (i,j) captures:

1. All squares in row i.
2. All squares in column j.
3. All squares on the diagonal i + j = a + b (the main diagonal direction).
4. All squares on the diagonal i - j = a - b (the anti-diagonal direction).

So, when we place a new piece, it must not be in any of these four types of squares for any existing piece.

But since the grid is huge, we can't check each square. We need a way to compute the total blocked squares by all existing pieces and subtract that from the total possible squares (N^2 - M), but wait, no. Because the problem says the new piece must be placed on an empty square. So the initial total possible squares is (N^2 - M). But some of these squares are blocked by the existing pieces. So the answer is the number of squares that are not occupied by existing pieces and are not captured by any existing piece.

Wait, no. The new piece is placed on an empty square. So the empty squares are all squares not in the existing pieces. So the initial count is (N^2 - M). But among these, some are under attack by existing pieces. So the answer is (N^2 - M) minus the number of empty squares that are under attack by any existing piece. Wait, no. Because the new square can't be under attack by any existing piece. So the answer is the number of empty squares that are not attacked by any existing piece.

So how to compute that?

Let me rephrase the problem: For each empty square (i,j), check if there's any existing piece that attacks (i,j). If not, then it's a valid square. The count of such squares is the answer.

But with N up to 1e9 and M up to 1e3, we can't check each square. So we need a mathematical approach to compute the number of squares that are not attacked by any existing piece and are empty.

Wait, but the existing pieces are already on the grid, so the empty squares are all squares not in the existing positions. So the problem reduces to: compute the number of squares not in the existing positions and not attacked by any existing piece.

So the steps are:

1. Compute the total number of squares not under attack by any existing piece.

2. Subtract the number of existing pieces that are in those squares (since those squares are already occupied and thus not available).

Wait, no. Because existing pieces are already in their positions, and the new piece can't be placed there. So the available squares are those that are:

a) Not occupied by any existing piece.

b) Not attacked by any existing piece.

So the answer is the number of squares that are not in the existing positions and are not attacked by any existing piece.

So the approach is:

- Compute the set of all squares attacked by existing pieces. Then, subtract the existing pieces from this set (since they are already occupied), and the result is the squares that are either occupied or attacked. Then, the answer is (N^2) - (size of the attacked set) - (M - size of existing pieces in attacked set) ? Wait, maybe not. Let's think again.

Alternatively, the total squares that are under attack by any existing piece is the union of all squares attacked by each existing piece. Then, the squares that are available are the ones not in this union and not occupied by existing pieces.

Wait, the existing pieces are part of the attacked squares (since each existing piece's own square is in their row, column, etc.), but the existing pieces are already occupied, so they are excluded from the available squares.

So the available squares are:

Total squares = N^2

Subtract the number of squares that are either occupied or under attack.

Wait, no. Because the available squares are the ones that are not occupied AND not under attack.

So the formula is:

Available = (Total squares not occupied) AND (not under attack by any existing piece).

Which is:

Available = (N^2 - M) - (number of squares that are not occupied but are under attack).

But how to compute the number of squares that are under attack and not occupied.

Alternatively, the number of squares under attack is the union of all squares attacked by existing pieces. Then, the available squares are those not in the union and not in the existing pieces.

But the existing pieces are part of the union, since each existing piece's own square is in their row, column, etc. So the union includes all existing pieces. So the available squares are (N^2 - size of the union) + (number of existing pieces not in the union) ? No, because existing pieces are in the union. So the union includes all squares attacked by any existing piece, including the existing pieces themselves.

So the available squares are the squares not in the union. Because:

- The union includes all squares under attack (including existing pieces).

- So, squares not in the union are not under attack, and thus can be placed, but they must also not be occupied. Wait, but if a square is not in the union, it's not under attack, but it could be occupied by an existing piece. Wait, but the existing pieces are part of the union. Because each existing piece's square is in their own row, column, etc. So the union includes all existing pieces. So the squares not in the union are not under attack and not occupied. Therefore, the available squares are exactly the squares not in the union.

Wait, let's see:

Suppose we have an existing piece at (a,b). Then, the union includes (a,b) because it's in row a, column b, etc. So all existing pieces are in the union. Therefore, the squares not in the union are the squares that are not under attack by any existing piece and are not occupied by any existing piece. Therefore, the answer is the number of squares not in the union.

So the problem reduces to computing the size of the union of all squares attacked by existing pieces, and subtracting that from N^2, then adding the number of existing pieces that are not in the union (but since all existing pieces are in the union, that's zero). Wait, no. Because the existing pieces are in the union. So the answer is N^2 - (size of the union).

Wait, let's take an example. Suppose N=8, M=6 as in sample input 1. The output is 2. Let's compute the union of all attacked squares. The sample output says that the available squares are 2, which implies that the union's size is 8*8 - 2 = 62. But in the sample input, there are 6 existing pieces. So 64 - 62 = 2. So yes, the answer is N^2 - size of the union.

So the formula is answer = N^2 - size of the union of all attacked squares.

But how to compute the size of the union?

The problem now is to compute the size of the union of all the rows, columns, diagonals, and anti-diagonals of the existing pieces.

But each existing piece contributes four lines (row, column, diagonal, anti-diagonal) to the union. The union is the set of all squares that are in any of these lines for any existing piece.

But computing the union size when M is up to 1e3 is manageable if we can model the lines and compute their overlaps.

So the approach is:

- For each existing piece, collect all the rows, columns, diagonals (sum i+j), and anti-diagonals (difference i-j).

- Then, the total number of squares in the union is the sum of the sizes of each of these lines, minus the overlaps (using inclusion-exclusion). But with M up to 1e3, this approach would be O(M^2), which is 1e6 operations, manageable.

But how to model the lines.

Each line can be represented as:

- A row line: all squares in row a. The size is N.

- A column line: all squares in column b. The size is N.

- A diagonal line (sum s = a + b): the number of squares on this diagonal is min(s-1, 2N - s + 1). Wait, for sum s, the diagonal starts at (1, s-1) when s-1 <= N, but as s increases, the diagonal shifts. The length of the diagonal is min(s-1, 2N - (s-1)) ? Wait, maybe the number of squares on the diagonal s = i + j is:

For s from 2 to 2N:

The number of squares is s-1 if s <= N+1, else 2N - (s-1) + 1? Wait, let's think:

For example, when N=8:

sum=2: (1,1) → 1 square.

sum=3: (1,2), (2,1) → 2 squares.

...

sum=9: (1,8), (2,7), ..., (8,1) → 8 squares.

sum=10: (2,8), (3,7), ..., (8,2) → 7 squares.

sum=16: (8,8) → 1 square.

So for sum s, the number of squares is:

if s <= N+1: s-1

else: 2N - (s-1) → 2N - s + 1.

Wait, for s=9 (N=8), 2*8 -9 +1 = 16 -9 +1=8, which is correct.

Yes. So the number of squares in a diagonal with sum s is:

min(s-1, 2*N - s + 1) → but wait, 2*N - s +1 is equivalent to (2N +1 - s).

Alternatively, the formula is:

number of squares on diagonal s (i+j=s) is:

max(0, min(s-1, 2*N) - max(1, s - N) + 1 )

Wait, maybe another approach. For a given s, the minimum i is max(1, s - N), since j = s - i, and j must be <= N. So i >= s - N. Also, i >=1. So i ranges from max(1, s - N) to min(N, s-1). The number of i's is min(N, s-1) - max(1, s - N) + 1.

But perhaps there's a simpler way. For s in 2 to 2N:

The number of squares is:

if s <= N+1: s-1

else: 2N - s +1.

So for s=2, 1.

s=3, 2.

...

s=N+1: N.

s>N+1: 2N - (s-1) → 2N - s +1.

So for example, N=8, s=10: 2*8 -10 +1= 7.

Yes.

Similarly, for the anti-diagonal (i - j = d). The number of squares on this diagonal is:

The diagonal can be represented by d = i - j.

The possible values of d range from -(N-1) to N-1.

For a given d, the number of squares is N - |d|.

Because, for example, d=0: the main anti-diagonal, N squares.

d=1: starts at (2,1), (3,2), ..., (N, N-1) → N-1 squares.

d=-1: (1,2), (2,3), ..., (N-1, N) → N-1 squares.

So for a given d, the number of squares is N - |d|.

So, for each existing piece (a, b), the four lines are:

Row a: size N.

Column b: size N.

Diagonal sum s = a + b: size S = (s <= N+1) ? s-1 : 2N - s +1.

Anti-diagonal difference d = a - b: size D = N - |d|.

But when we collect all these lines, the total squares in the union is the sum of the sizes of all unique lines (row, column, sum, difference) minus the overlaps between them.

But how to compute the union size.

The problem is similar to the classic inclusion-exclusion problem where you have multiple sets and you need the size of their union.

But with four types of sets per existing piece, and M up to 1e3, the total number of lines is 4*M. But each line is a row, column, sum, or difference. So we can group them by their type and their identifier (like for rows, the identifier is a, for sum it's s, etc.)

So the idea is:

- For all existing pieces, collect all the rows, columns, sum diagonals, and difference diagonals.

- For each group (rows, columns, sums, differences), keep track of the unique identifiers (e.g., for rows, the identifiers are the a values).

- Then, the total number of squares covered by rows is the number of unique rows multiplied by N.

- Similarly for columns: number of unique columns multiplied by N.

- For sum diagonals: sum over all unique sum values s of the number of squares in s.

- For difference diagonals: sum over all unique difference values d of the number of squares in d.

But then, we have overlaps between these groups. For example, a square could be covered by a row and a column, or a row and a sum diagonal, etc.

So the inclusion-exclusion principle must be applied to compute the correct union size.

The formula for the union of four sets (rows, columns, sums, differences) is:

size(union) = size(rows) + size(columns) + size(sums) + size(differences) 

minus size(rows ∩ columns) + size(rows ∩ sums) + size(rows ∩ differences) + size(columns ∩ sums) + size(columns ∩ differences) + size(sums ∩ differences) 

plus size(rows ∩ columns ∩ sums) + ... all triple intersections 

minus size(rows ∩ columns ∩ sums ∩ differences)

But this seems very complicated, especially with M up to 1e3. However, considering that each of these intersections can be computed by the overlaps between the different line types, perhaps there's a smarter way.

Alternatively, since each line is a row, column, sum, or difference, the squares covered by any of the lines are those that are in any row, any column, any sum diagonal, or any difference diagonal of the existing pieces.

But the problem is to compute the union of all these lines.

An alternative approach is to model the union as the set of all squares that are in any of the rows, columns, sum diagonals, or difference diagonals of the existing pieces.

So the total number of squares in the union is:

sum_rows + sum_columns + sum_sum_diags + sum_diff_diags 

- sum_row_and_column_intersections 

- sum_row_and_sum_diag_intersections 

- sum_row_and_diff_diag_intersections 

- sum_column_and_sum_diag_intersections 

- sum_column_and_diff_diag_intersections 

- sum_sum_diag_and_diff_diag_intersections 

+ sum_row_and_column_and_sum_diag_intersections 

+ sum_row_and_column_and_diff_diag_intersections 

+ sum_row_and_sum_diag_and_diff_diag_intersections 

+ sum_column_and_sum_diag_and_diff_diag_intersections 

- sum_all_four_intersections 

+ ... and so on.

But this inclusion-exclusion for four sets is computationally expensive. However, perhaps the overlapping between different line types can be calculated in a manageable way.

Alternatively, perhaps the overlapping between different line types can be considered as the squares that are in the intersection of two or more lines from different types. For example, a square that is in a row of one piece and a column of another piece. But how to compute the total number of such squares across all possible pairs.

But given that M is up to 1e3, the number of rows, columns, sum diagonals, and difference diagonals can be up to 1e3 each. So the total unique rows is up to 1e3, same for columns, sum diagonals, and difference diagonals.

So let's denote:

R = set of all rows (a) from existing pieces.

C = set of all columns (b) from existing pieces.

S = set of all sum diagonals (a + b) from existing pieces.

D = set of all difference diagonals (a - b) from existing pieces.

Then, the total squares covered by rows is |R| * N.

Similarly for columns: |C| * N.

For sum diagonals: sum_{s in S} (s <= N+1 ? s-1 : 2N - s +1).

For difference diagonals: sum_{d in D} (N - |d|).

Then, the overlaps between rows and columns: for each row in R and column in C, the intersection is the square (row, column). So the number of such squares is |R| * |C|. But wait, no. The intersection of a row and a column is a single square. So the total number of squares in the intersection of rows and columns is the number of pairs (r, c) where r is in R and c is in C. Each such pair is a single square. So the count is |R| * |C|.

Similarly, the intersection between a row and a sum diagonal: for a row r and a sum s, the squares that are in row r and sum s. The sum s = r + j, where j is the column. So j = s - r. So in row r, the column j must be s - r, and also j must be between 1 and N. So for each (r, s), there is at most one square in the intersection. So the number of squares in the intersection of rows and sum diagonals is the number of pairs (r, s) where r is in R, s is in S, and 1 <= s - r <= N. For each such pair, there is one square. So the count is the number of (r, s) pairs in R x S where s - r is between 1 and N.

Similarly for other intersections.

But this approach would require checking all possible pairs between the different sets, which could be up to (1e3)^2 = 1e6 operations for each pair of sets, which is manageable.

So the steps are:

1. Compute R, C, S, D.

2. Compute the sum of each individual set's contribution.

3. Compute the sum of all pairwise intersections.

4. Compute the sum of all triple intersections.

5. Compute the sum of all four-way intersections.

6. Apply inclusion-exclusion to get the total union size.

But this seems feasible.

Let me outline the steps in code.

First, collect R, C, S, D:

For each piece (a, b):

- Add a to R.

- Add b to C.

- Add (a + b) to S.

- Add (a - b) to D.

Then, R, C, S, D are sets (so duplicates are removed).

Then, compute:

sum_rows = len(R) * N

sum_cols = len(C) * N

sum_sum = sum( (s-1) if s <= N+1 else (2*N - s +1) for s in S )

sum_diff = sum( N - abs(d) for d in D )

total_individual = sum_rows + sum_cols + sum_sum + sum_diff

Then, compute pairwise intersections:

pair_row_col = len(R) * len(C)

pair_row_sum = sum(1 for r in R for s in S if 1 <= (s - r) <= N )

pair_row_diff = sum(1 for r in R for d in D if 1 <= (r + d) <= N )

pair_col_sum = sum(1 for c in C for s in S if 1 <= (s - c) <= N )

pair_col_diff = sum(1 for c in C for d in D if 1 <= (c - d) <= N )

pair_sum_diff = sum(1 for s in S for d in D if ( (s - d) % 2 == 0 ) and ( (s + d) // 2 >= 1 ) and ( (s - d) // 2 >= 1 ) and ( (s + d) // 2 <= N ) and ( (s - d) // 2 <= N ) )

Wait, this last part is the intersection between sum diagonal s and difference diagonal d.

A square (i,j) is on sum s = i + j and difference d = i - j.

So solving for i and j:

i + j = s

i - j = d

Adding: 2i = s + d → i = (s + d)/2

j = (s - d)/2

So i and j must be integers between 1 and N.

So for the intersection of sum s and difference d, there is a square (i,j) if and only if:

(s + d) is even,

i = (s + d)/2 is between 1 and N,

j = (s - d)/2 is between 1 and N.

So the count for pair_sum_diff is the number of (s, d) pairs in S x D where (s + d) is even, and 1 <= (s + d)/2 <= N, and 1 <= (s - d)/2 <= N.

So that's the calculation.

So sum_pair_second is the sum of all these pairwise intersections.

Then, the total pairwise overlaps is:

pair_row_col + pair_row_sum + pair_row_diff + pair_col_sum + pair_col_diff + pair_sum_diff

Then, the triple overlaps:

For example, row, column, sum: the square (r, c) must be in row r, column c, and sum s = r + c. So for a triple (r, c, s), the square (r, c) must be in R x C x S. So the number of such squares is the number of triples (r, c, s) where r is in R, c is in C, s is in S, and r + c = s.

Similarly for other triples.

But this seems complicated. For each triple (row, column, sum), check if r + c = s.

The count for this triple overlap is the number of (r, c, s) in R x C x S where r + c = s.

Similarly:

triple_row_col_sum = sum(1 for r in R for c in C for s in S if r + c == s )

triple_row_col_diff = sum(1 for r in R for c in C for d in D if r - c == d )

triple_row_sum_diff = sum(1 for r in R for s in S for d in D if ... )

Wait, for triple_row_sum_diff: the square is in row r, sum s, and difference d. So:

i = r

i + j = s → j = s - r

Also, i - j = d → r - (s - r) = d → 2r - s = d → s = 2r - d.

So the square (r, s - r) must satisfy:

j = s - r >=1 and <= N

and also, d = r - j = r - (s - r) = 2r - s.

But since s is in S and d is in D, the triple overlap is the number of (r, s, d) where r is in R, s is in S, d is in D, and 2r - s = d, and 1 <= s - r <= N.

But this seems complicated. For each (r, s, d) in R x S x D, check if 2r - s = d and j = s - r is between 1 and N.

But this is O(1e3^3) operations, which is 1e9, which is way too slow.

This approach is not feasible for M=1e3, as triple overlaps would require 1e9 operations.

So this suggests that the inclusion-exclusion approach is not feasible for large M.

Hmm. So we need a smarter way.

Alternative approach: the inclusion-exclusion for four sets is computationally infeasible. So perhaps the problem can be transformed into a mathematical model that allows us to compute the union size without explicitly considering all overlaps.

Wait, let's think again: each square in the union is in at least one of the rows, columns, sum diagonals, or difference diagonals of the existing pieces.

But each square can be in multiple lines. For example, a square (r, c) is in row r, column c, sum r + c, and difference r - c.

But the problem is to count the number of squares that are in any of the lines (row, column, sum, difference) of any existing piece.

So the square (i,j) is in the union if:

i is in R (rows of existing pieces),

OR j is in C (columns of existing pieces),

OR (i + j) is in S (sum diagonals of existing pieces),

OR (i - j) is in D (difference diagonals of existing pieces).

So the total number of squares in the union is the number of (i,j) in [1..N]^2 that satisfy i ∈ R OR j ∈ C OR (i+j) ∈ S OR (i-j) ∈ D.

We need to compute the size of this set.

But how to compute this efficiently.

The key insight is that the problem can be transformed into a 4-dimensional inclusion-exclusion, but perhaps there's a way to compute the count by breaking it down into parts.

Another approach is to model the problem as:

Number of squares in union = (number of squares in R rows) + (number of squares in C columns) + (number of squares in S sum diagonals) + (number of squares in D difference diagonals) 

- (number of squares in both R rows and C columns) 

- (number of squares in R rows and S sum diagonals) 

- (number of squares in R rows and D difference diagonals) 

- (number of squares in C columns and S sum diagonals) 

- (number of squares in C columns and D difference diagonals) 

- (number of squares in S sum diagonals and D difference diagonals) 

+ (number of squares in R rows, C columns, and S sum diagonals) 

+ (number of squares in R rows, C columns, and D difference diagonals) 

+ (number of squares in R rows, S sum diagonals, and D difference diagonals) 

+ (number of squares in C columns, S sum diagonals, and D difference diagonals) 

- (number of squares in R rows, C columns, S sum diagonals, and D difference diagonals) 

But again, this requires computing all these terms, which may be feasible for M=1e3.

But how to compute each term.

Let's proceed step by step.

First, compute the individual terms:

A = |R| * N

B = |C| * N

C = sum of squares in S sum diagonals → sum_{s in S} (s-1 if s <= N+1 else 2N - s +1)

D = sum of squares in D difference diagonals → sum_{d in D} (N - |d|)

Total individual = A + B + C + D

Now, subtract the pairwise overlaps:

Pairwise overlaps:

AB: squares that are in R rows and C columns → each such square is (r, c) where r is in R and c is in C. The count is |R| * |C|.

AC: squares in R rows and S sum diagonals → for each r in R, s in S, the square (r, s - r) must be valid (j = s - r between 1 and N). So for each (r, s), if 1 <= s - r <= N, then this square is counted. So the count is the number of (r, s) pairs where r is in R, s is in S, and 1 <= s - r <= N.

AD: squares in R rows and D difference diagonals → for each r in R, d in D, the square is (r, r - d). j = r - d must be between 1 and N. So count the number of (r, d) pairs where r is in R, d in D, and 1 <= r - d <= N.

BC: squares in C columns and S sum diagonals → for each c in C, s in S, the square is (s - c, c). i = s - c must be between 1 and N. So count the number of (c, s) pairs where c is in C, s in S, and 1 <= s - c <= N.

BD: squares in C columns and D difference diagonals → for each c in C, d in D, the square is (c + d, c). i = c + d must be between 1 and N. So count the number of (c, d) pairs where c in C, d in D, and 1 <= c + d <= N.

CD: squares in S sum diagonals and D difference diagonals → for each s in S, d in D, the square (i,j) must satisfy i + j = s and i - j = d. Solving, i = (s + d)/2, j = (s - d)/2. So i and j must be integers between 1 and N. So count the number of (s, d) pairs where s in S, d in D, (s + d) is even, and 1 <= (s + d)/2 <= N, and 1 <= (s - d)/2 <= N.

So pairwise overlaps total is AB + AC + AD + BC + BD + CD.

Then, add back the triple overlaps:

ABC: squares in R rows, C columns, and S sum diagonals → this is the number of (r, c) in R x C where r + c is in S. Because the square (r, c) is in row r, column c, and sum diagonal r + c. So for each (r, c) in R x C, check if r + c is in S. The count is the number of (r, c) pairs in R x C such that r + c is in S.

ABD: squares in R rows, C columns, and D difference diagonals → for each (r, c) in R x C, check if r - c is in D. The count is the number of (r, c) pairs in R x C such that r - c is in D.

ACD: squares in R rows, S sum diagonals, and D difference diagonals → for each (r, s, d) in R x S x D, the square (r, j) must be in row r, sum s, and difference d. So j = s - r. Also, difference d = r - j = r - (s - r) = 2r - s. So for a given r, s, d, check if 2r - s = d. Also, j must be between 1 and N. So the square is (r, s - r), and this must also be in difference diagonal d. So the count is the number of (r, s, d) in R x S x D where 2r - s = d and 1 <= s - r <= N.

But this is O(1e3^3) operations, which is 1e9, which is not feasible.

Wait, but maybe there's a smarter way. For each r in R, s in S, compute d = 2r - s. If d is in D, then check if j = s - r is between 1 and N. So for each (r, s) in R x S, compute d and see if it's in D, and j is valid. So the count is the number of (r, s) in R x S where d = 2r - s is in D and j = s - r is between 1 and N.

This is O(1e6) operations, which is manageable.

Similarly, for other triple overlaps.

BCD: squares in C columns, S sum diagonals, and D difference diagonals → for each (c, s) in C x S, compute i = s - c, then check if i is in R? No, wait, the square is in column c, sum s, and difference d. So i = s - c, j = c. Then, the difference is i - j = (s - c) - c = s - 2c. So the difference d must be in D. So for each (c, s) in C x S, check if s - 2c is in D, and i = s - c is between 1 and N.

So the count is the number of (c, s) in C x S where (s - 2c) is in D and 1 <= s - c <= N.

So for BCD, count is the number of (c, s) in C x S where (s - 2c) is in D and 1 <= s - c <= N.

This is O(1e6) operations.

So for triple overlaps, each can be computed in O(1e6) steps.

Then, the triple overlaps total is ABC + ABD + ACD + BCD.

Finally, subtract the four-way overlaps:

ABCD: squares that are in R rows, C columns, S sum diagonals, and D difference diagonals → for a square (r, c) in R x C, it must also be in S and D. So r + c must be in S, and r - c must be in D. So the count is the number of (r, c) in R x C where (r + c) is in S and (r - c) is in D.

This is O(1e6) operations.

Putting it all together, the inclusion-exclusion formula is:

union_size = (A + B + C + D) 

- (AB + AC + AD + BC + BD + CD) 

+ (ABC + ABD + ACD + BCD) 

- (ABCD)

So the code needs to compute all these terms.

Now, let's think about how to compute each term.

First, collect R, C, S, D as sets.

Then:

Compute A = len(R) * N

B = len(C) * N

C = sum for each s in S: (s-1 if s <= N+1 else 2*N - s +1)

D = sum for each d in D: (N - abs(d))

AB = len(R) * len(C)

For AC: count the number of (r, s) in R x S where 1 <= s - r <= N.

In code: AC = sum(1 for r in R for s in S if 1 <= (s - r) <= N)

Similarly for AD: sum(1 for r in R for d in D if 1 <= (r - d) <= N )

Wait, AD is the squares in R rows and D difference diagonals. For a row r and difference d, the square is (r, j), where j = r - d. So j must be between 1 and N. So 1 <= r - d <= N → d >= r - N and d <= r -1.

But in code, for each r in R and d in D, check if (r - d) is between 1 and N.

So AD is sum(1 for r in R for d in D if 1 <= (r - d) <= N )

Similarly, BC is sum(1 for c in C for s in S if 1 <= (s - c) <= N )

BD is sum(1 for c in C for d in D if 1 <= (c + d) <= N )

CD is sum(1 for s in S for d in D if (s + d) % 2 == 0 and ( (s + d) // 2 >= 1 and (s + d) // 2 <= N ) and ( (s - d) // 2 >= 1 and (s - d) // 2 <= N ) )

Then, the triple overlaps:

ABC is sum(1 for r in R for c in C if (r + c) in S )

ABD is sum(1 for r in R for c in C if (r - c) in D )

ACD is sum(1 for r in R for s in S if (2*r - s) in D and 1 <= (s - r) <= N )

BCD is sum(1 for c in C for s in S if (s - 2*c) in D and 1 <= (s - c) <= N )

Finally, ABCD is sum(1 for r in R for c in C if (r + c) in S and (r - c) in D )

Now, all these terms can be computed with O(M^2) operations, except for the triple and four-way overlaps which may require O(M^3) steps. But wait, for example, ABC is O(M^2) because R and C are each up to M, and checking if r + c is in S can be done with a hash set. Similarly for others.

So the key is to precompute the sets R, C, S, D as hash sets for O(1) lookups.

So let's outline the code steps:

Read N, M.

Read M pairs (a, b). Collect R, C, S, D as sets.

Compute A, B, C, D.

Compute AB, AC, AD, BC, BD, CD.

Compute ABC, ABD, ACD, BCD.

Compute ABCD.

Then, union_size = A + B + C + D - (AB + AC + AD + BC + BD + CD) + (ABC + ABD + ACD + BCD) - ABCD

The answer is N*N - union_size.

But wait, let's test this with sample input 1.

Sample Input 1:

8 6

1 4 → R={1}, C={4}, S=5, D=-3

2 1 → R={1,2}, C={4,1}, S=3, D=1

3 8 → R={1,2,3}, C={4,1,8}, S=11, D=-5

4 5 → R={1,2,3,4}, C={4,1,8,5}, S=9, D=-1

5 2 → R={1,2,3,4,5}, C={4,1,8,5,2}, S=7, D=3

8 3 → R={1,2,3,4,5,8}, C={4,1,8,5,2,3}, S=11, D=5

So R = {1,2,3,4,5,8}

C = {1,2,3,4,5,8}

S = {5,3,11,9,7}

D = {-3,1,-5,-1,3,5}

Wait, let's compute S and D for each piece:

1st piece (1,4): s=5, d= -3 → added to S and D.

2nd (2,1): s=3, d=1 → added.

3rd (3,8): s=11, d= -5 → added.

4th (4,5): s=9, d= -1 → added.

5th (5,2): s=7, d=3 → added.

6th (8,3): s=11, d=5 → added.

So S = {3,5,7,9,11}

D = {-5, -3, -1, 1, 3,5}

Now, compute A = len(R) * 8 → 6 * 8 = 48.

B = len(C) * 8 → 6 * 8 = 48.

C is sum of s in S:

s=3: 3 <= 8+1=9 → 3-1=2.

s=5: 5-1=4.

s=7: 7-1=6.

s=9: 9 <=9 → 9-1=8.

s=11: 2*8 -11 +1 = 16 -11 +1=6.

So sum for C is 2+4+6+8+6=26.

D is sum of N - |d| for each d in D:

d=-5 → 8 -5=3.

d=-3 → 8-3=5.

d=-1 →8-1=7.

d=1 →8-1=7.

d=3 →8-3=5.

d=5 →8-5=3.

Sum: 3+5+7+7+5+3= 30.

So total_individual = 48 + 48 + 26 +30 = 152.

Now pairwise overlaps:

AB: 6 * 6 =36.

AC: for each r in R and s in S, check if 1 <= s - r <=8.

R is {1,2,3,4,5,8}, S is {3,5,7,9,11}.

For each r and s:

r=1:

s=3 → 3-1=2 → valid.

s=5 →4 → valid.

s=7 →6 → valid.

s=9 →8 → valid.

s=11 →10 → invalid (since N=8).

So 4 valid.

r=2:

s=3 →1 → valid.

s=5 →3 → valid.

s=7 →5 → valid.

s=9 →7 → valid.

s=11 →9 → invalid.

4 valid.

r=3:

s=3 →0 → invalid.

s=5 →2 → valid.

s=7 →4 → valid.

s=9 →6 → valid.

s=11 →8 → valid.

4 valid.

r=4:

s=3 →-1 → invalid.

s=5 →1 → valid.

s=7 →3 → valid.

s=9 →5 → valid.

s=11 →7 → valid.

4 valid.

r=5:

s=3 →-2 → invalid.

s=5 →0 → invalid.

s=7 →2 → valid.

s=9 →4 → valid.

s=11 →6 → valid.

3 valid.

r=8:

s=3 →-5 → invalid.

s=5 →-3 → invalid.

s=7 →-1 → invalid.

s=9 →1 → valid.

s=11 →3 → valid.

2 valid.

Total AC: 4+4+4+4+3+2= 21.

AD: for each r in R and d in D, check if 1 <= r -d <=8.

D is {-5, -3, -1, 1, 3,5}.

For each r in R:

r=1:

d=-5 →1 - (-5) =6 → valid.

d=-3 →4 → valid.

d=-1 →2 → valid.

d=1 →0 → invalid.

d=3 →-2 → invalid.

d=5 →-4 → invalid.

So 3 valid.

r=2:

d=-5 →7 → valid.

d=-3 →5 → valid.

d=-1 →3 → valid.

d=1 →1 → valid.

d=3 →-1 → invalid.

d=5 →-3 → invalid.

4 valid.

r=3:

d=-5 →8 → valid.

d=-3 →6 → valid.

d=-1 →4 → valid.

d=1 →2 → valid.

d=3 →0 → invalid.

d=5 →-2 → invalid.

4 valid.

r=4:

d=-5 →9 → invalid (since N=8).

d=-3 →7 → valid.

d=-1 →5 → valid.

d=1 →3 → valid.

d=3 →1 → valid.

d=5 →-1 → invalid.

4 valid.

r=5:

d=-5 →10 → invalid.

d=-3 →8 → valid.

d=-1 →6 → valid.

d=1 →4 → valid.

d=3 →2 → valid.

d=5 →0 → invalid.

4 valid.

r=8:

d=-5 →13 → invalid.

d=-3 →11 → invalid.

d=-1 →9 → invalid.

d=1 →7 → valid.

d=3 →5 → valid.

d=5 →3 → valid.

3 valid.

Total AD: 3+4+4+4+4+3=22.

BC: for each c in C and s in S, check if 1 <= s -c <=8.

C is {1,2,3,4,5,8}.

For each c:

c=1:

s=3 →2 → valid.

s=5 →4 → valid.

s=7 →6 → valid.

s=9 →8 → valid.

s=11 →10 → invalid.

4 valid.

c=2:

s=3 →1 → valid.

s=5 →3 → valid.

s=7 →5 → valid.

s=9 →7 → valid.

s=11 →9 → invalid.

4 valid.

c=3:

s=3 →0 → invalid.

s=5 →2 → valid.

s=7 →4 → valid.

s=9 →6 → valid.

s=11 →8 → valid.

4 valid.

c=4:

s=3 →-1 → invalid.

s=5 →1 → valid.

s=7 →3 → valid.

s=9 →5 → valid.

s=11 →7 → valid.

4 valid.

c=5:

s=3 →-2 → invalid.

s=5 →0 → invalid.

s=7 →2 → valid.

s=9 →4 → valid.

s=11 →6 → valid.

3 valid.

c=8:

s=3 →-5 → invalid.

s=5 →-3 → invalid.

s=7 →-1 → invalid.

s=9 →1 → valid.

s=11 →3 → valid.

2 valid.

Total BC: 4+4+4+4+3+2=21.

BD: for each c in C and d in D, check if 1 <= c +d <=8.

D is {-5, -3, -1, 1, 3,5}.

For each c:

c=1:

d=-5 →-4 → invalid.

d=-3 →-2 → invalid.

d=-1 →0 → invalid.

d=1 →2 → valid.

d=3 →4 → valid.

d=5 →6 → valid.

3 valid.

c=2:

d=-5 →-3 → invalid.

d=-3 →-1 → invalid.

d=-1 →1 → valid.

d=1 →3 → valid.

d=3 →5 → valid.

d=5 →7 → valid.

4 valid.

c=3:

d=-5 →-2 → invalid.

d=-3 →0 → invalid.

d=-1 →2 → valid.

d=1 →4 → valid.

d=3 →6 → valid.

d=5 →8 → valid.

4 valid.

c=4:

d=-5 →-1 → invalid.

d=-3 →1 → valid.

d=-1 →3 → valid.

d=1 →5 → valid.

d=3 →7 → valid.

d=5 →9 → invalid.

4 valid.

c=5:

d=-5 →0 → invalid.

d=-3 →2 → valid.

d=-1 →4 → valid.

d=1 →6 → valid.

d=3 →8 → valid.

d=5 →10 → invalid.

4 valid.

c=8:

d=-5 →3 → valid.

d=-3 →5 → valid.

d=-1 →7 → valid.

d=1 →9 → invalid.

d=3 →11 → invalid.

d=5 →13 → invalid.

3 valid.

Total BD: 3+4+4+4+4+3=22.

CD: for each s in S and d in D, check if (s +d) is even, and (s +d)/2 and (s -d)/2 are between 1 and 8.

S is {3,5,7,9,11}, D is {-5, -3, -1,1,3,5}.

For each (s, d):

s=3:

d=-5 → 3 + (-5) =-2 → even? -2 is even. i= (-2)/2 =-1 → invalid.

d=-3 →0 → even. i=0 → invalid.

d=-1 →2 → even. i=1, j= (3 - (-1))/2=2 → (1,2). valid.

d=1 →4 → even. i=2, j=1 → valid.

d=3 →6 → even. i=3, j=0 → invalid.

d=5 →8 → even. i=4, j= (3-5)/2= -1 → invalid.

So for s=3, valid d are -1 and 1 → 2.

s=5:

d=-5 →0 → even. i=0 → invalid.

d=-3 →2 → even. i=1, j= (5 - (-3))/2=4 → (1,4). valid.

d=-1 →4 → even. i=2, j=3 → valid.

d=1 →6 → even. i=3, j=2 → valid.

d=3 →8 → even. i=4, j=1 → valid.

d=5 →10 → even. i=5, j=0 → invalid.

So 4 valid.

s=7:

d=-5 →2 → even. i= (7-5)/2=1, j= (7 +5)/2=6 → (1,6). valid.

Wait, wait, solving for i and j:

i = (s +d)/2 = (7-5)/2 = 1.

j = (s -d)/2 = (7 +5)/2 =6.

So i=1, j=6 → valid.

d=-3 →4 → even. i= (7-3)/2=2, j= (7+3)/2=5 → valid.

d=-1 →6 → even. i= (7-1)/2=3, j=4 → valid.

d=1 →8 → even. i=4, j=3 → valid.

d=3 →10 → even. i=5, j=2 → valid.

d=5 →12 → even. i=6, j=1 → valid.

So all d for s=7:

d=-5 → valid.

d=-3 → valid.

d=-1 → valid.

d=1 → valid.

d=3 → valid.

d=5 → valid.

So 6 valid.

s=9:

d=-5 →4 → even. i= (9-5)/2=2, j= (9+5)/2=7 → valid.

d=-3 →6 → even. i=3, j=6 → valid.

d=-1 →8 → even. i=4, j=5 → valid.

d=1 →10 → even. i=5, j=4 → valid.

d=3 →12 → even. i=6, j=3 → valid.

d=5 →14 → even. i=7, j=2 → valid.

So 6 valid.

s=11:

d=-5 →6 → even. i= (11-5)/2=3, j= (11+5)/2=8 → valid.

d=-3 →8 → even. i=4, j=7 → valid.

d=-1 →10 → even. i=5, j=6 → valid.

d=1 →12 → even. i=6, j=5 → valid.

d=3 →14 → even. i=7, j=4 → valid.

d=5 →16 → even. i=8, j=3 → valid.

So 6 valid.

Total CD: 2 +4 +6 +6 +6 = 24.

So pairwise overlaps total is AB (36) + AC (21) + AD (22) + BC (21) + BD (22) + CD (24) → 36+21=57; 57+22=79; 79+21=100; 100+22=122; 122+24=146.

So subtract 146 from 152 → 152 -146 = 6.

Now, triple overlaps:

ABC: sum(1 for r in R for c in C if (r + c) in S).

R is {1,2,3,4,5,8}, C is {1,2,3,4,5,8}, S is {3,5,7,9,11}.

For each (r, c) in R x C:

Check if r + c is in S.

For example:

r=1, c=1 →2 → not in S.

r=1, c=2 →3 → yes.

r=1, c=3 →4 → no.

r=1, c=4 →5 → yes.

r=1, c=5 →6 → no.

r=1, c=8 →9 → yes.

So for r=1, 3 valid c's.

Similarly, this would take time. Let's think of how many (r,c) pairs sum to a value in S.

But this is O(6*6)=36 operations.

But let's compute this:

S = {3,5,7,9,11}

Possible sums:

r + c can be from 1+1=2 to 8+8=16.

We need to count how many (r,c) pairs in R x C have sum in S.

R and C are {1,2,3,4,5,8}.

Possible sums:

For r=1:

c=1 →2 (no)

c=2 →3 (yes)

c=3 →4 (no)

c=4 →5 (yes)

c=5 →6 (no)

c=8 →9 (yes)

→ 3.

r=2:

c=1 →3 (yes)

c=2 →4 (no)

c=3 →5 (yes)

c=4 →6 (no)

c=5 →7 (yes)

c=8 →10 (no)

→3.

r=3:

c=1 →4 (no)

c=2 →5 (yes)

c=3 →6 (no)

c=4 →7 (yes)

c=5 →8 (no)

c=8 →11 (yes)

→3.

r=4:

c=1 →5 (yes)

c=2 →6 (no)

c=3 →7 (yes)

c=4 →8 (no)

c=5 →9 (yes)

c=8 →12 (no)

→3.

r=5:

c=1 →6 (no)

c=2 →7 (yes)

c=3 →8 (no)

c=4 →9 (yes)

c=5 →10 (no)

c=8 →13 (no)

→2.

r=8:

c=1 →9 (yes)

c=2 →10 (no)

c=3 →11 (yes)

c=4 →12 (no)

c=5 →13 (no)

c=8 →16 (no)

→2.

Total ABC: 3+3+3+3+2+2=16.

ABD: sum(1 for r in R for c in C if (r - c) in D).

D is {-5, -3, -1, 1, 3,5}.

For each (r,c) in R x C:

r - c must be in D.

For example:

r=1, c=1 →0 → no.

r=1, c=2 →-1 → yes.

r=1, c=3 →-2 → no.

r=1, c=4 →-3 → yes.

r=1, c=5 →-4 → no.

r=1, c=8 →-7 → no.

So for r=1, 2 valid.

r=2:

c=1 →1 → yes.

c=2 →0 → no.

c=3 →-1 → yes.

c=4 →-2 → no.

c=5 →-3 → yes.

c=8 →-6 → no.

3 valid.

r=3:

c=1 →2 → no.

c=2 →1 → yes.

c=3 →0 → no.

c=4 →-1 → yes.

c=5 →-2 → no.

c=8 →-5 → yes.

3 valid.

r=4:

c=1 →3 → yes.

c=2 →2 → no.

c=3 →1 → yes.

c=4 →0 → no.

c=5 →-1 → yes.

c=8 →-4 → no.

3 valid.

r=5:

c=1 →4 → no.

c=2 →3 → yes.

c=3 →2 → no.

c=4 →1 → yes.

c=5 →0 → no.

c=8 →-3 → yes.

3 valid.

r=8:

c=1 →7 → no.

c=2 →6 → no.

c=3 →5 → yes.

c=4 →4 → no.

c=5 →3 → yes.

c=8 →0 → no.

2 valid.

Total ABD: 2+3+3+3+3+2=16.

ACD: sum(1 for r in R for s in S if (2*r - s) in D and 1 <= (s - r) <=8 )

For each r in R and s in S:

compute d = 2*r - s.

Check if d is in D, and j = s - r is between 1 and 8.

For example:

r=1, s=3: d=2*1 -3 =-1 → in D. j=3-1=2 → valid. → count.

r=1, s=5: d=2-5=-3 → in D. j=4 → valid.

r=1, s=7: d=2-7=-5 → in D. j=6 → valid.

r=1, s=9: d=2-9=-7 → not in D. j=8 → valid.

So for r=1, s=3,5,7 are valid.

So 3.

r=1, s=11: d=2-11=-9 → no.

So for r=1, 3 valid.

r=2, s=3: d=4-3=1 → in D. j=1 → valid.

r=2, s=5: d=4-5=-1 → in D. j=3 → valid.

r=2, s=7: d=4-7=-3 → in D. j=5 → valid.

r=2, s=9: d=4-9=-5 → in D. j=7 → valid.

r=2, s=11: d=4-11=-7 → no. j=9 → invalid.

So 4 valid.

r=3, s=3: d=6-3=3 → in D. j=0 → invalid.

r=3, s=5: d=6-5=1 → in D. j=2 → valid.

r=3, s=7: d=6-7=-1 → in D. j=4 → valid.

r=3, s=9: d=6-9=-3 → in D. j=6 → valid.

r=3, s=11: d=6-11=-5 → in D. j=8 → valid.

So 4 valid.

r=4, s=3: d=8-3=5 → in D. j=3-4=-1 → invalid.

r=4, s=5: d=8-5=3 → in D. j=1 → valid.

r=4, s=7: d=8-7=1 → in D. j=3 → valid.

r=4, s=9: d=8-9=-1 → in D. j=5 → valid.

r=4, s=11: d=8-11=-3 → in D. j=7 → valid.

So 4 valid.

r=5, s=3: d=10-3=7 → not in D.

r=5, s=5: d=10-5=5 → in D. j=0 → invalid.

r=5, s=7: d=10-7=3 → in D. j=2 → valid.

r=5, s=9: d=10-9=1 → in D. j=4 → valid.

r=5, s=11: d=10-11=-1 → in D. j=6 → valid.

So 3 valid.

r=8, s=3: d=16-3=13 → no.

r=8, s=5: d=16-5=11 → no.

r=8, s=7: d=16-7=9 → no.

r=8, s=9: d=16-9=7 → no.

r=8, s=11: d=16-11=5 → in D. j=11-8=3 → valid.

So 1 valid.

Total ACD: 3+4+4+4+3+1=19.

BCD: sum(1 for c in C for s in S if (s - 2*c) in D and 1 <= (s - c) <=8 )

For each c in C and s in S:

d = s - 2*c.

Check if d is in D, and i = s - c is between 1 and 8.

For example:

c=1, s=3: d=3-2*1=1 → in D. i=3-1=2 → valid.

c=1, s=5: d=5-2=3 → in D. i=4 → valid.

c=1, s=7: d=7-2=5 → in D. i=6 → valid.

c=1, s=9: d=9-2=7 → no. i=8 → valid.

So 3 valid.

c=1, s=11: d=11-2=9 → no.

c=2, s=3: d=3-4=-1 → in D. i=1 → valid.

c=2, s=5: d=5-4=1 → in D. i=3 → valid.

c=2, s=7: d=7-4=3 → in D. i=5 → valid.

c=2, s=9: d=9-4=5 → in D. i=7 → valid.

c=2, s=11: d=11-4=7 → no. i=9 → invalid.

So 4 valid.

c=3, s=3: d=3-6=-3 → in D. i=0 → invalid.

c=3, s=5: d=5-6=-1 → in D. i=2 → valid.

c=3, s=7: d=7-6=1 → in D. i=4 → valid.

c=3, s=9: d=9-6=3 → in D. i=6 → valid.

c=3, s=11: d=11-6=5 → in D. i=8 → valid.

So 4 valid.

c=4, s=3: d=3-8=-5 → in D. i=-1 → invalid.

c=4, s=5: d=5-8=-3 → in D. i=1 → valid.

c=4, s=7: d=7-8=-1 → in D. i=3 → valid.

c=4, s=9: d=9-8=1 → in D. i=5 → valid.

c=4, s=11: d=11-8=3 → in D. i=7 → valid.

So 4 valid.

c=5, s=3: d=3-10=-7 → no.

c=5, s=5: d=5-10=-5 → in D. i=0 → invalid.

c=5, s=7: d=7-10=-3 → in D. i=2 → valid.

c=5, s=9: d=9-10=-1 → in D. i=4 → valid.

c=5, s=11: d=11-10=1 → in D. i=6 → valid.

So 3 valid.

c=8, s=3: d=3-16=-13 → no.

c=8, s=5: d=5-16=-11 → no.

c=8, s=7: d=7-16=-9 → no.

c=8, s=9: d=9-16=-7 → no.

c=8, s=11: d=11-16=-5 → in D. i=3 → valid.

So 1 valid.

Total BCD: 3+4+4+4+3+1=19.

So triple overlaps total is 16 +16 +19 +19 = 50.

So add 50 → 6 +50 =56.

Now, subtract ABCD: sum(1 for r in R for c in C if (r + c) in S and (r - c) in D).

For each (r,c) in R x C:

Check if (r + c) is in S and (r - c) is in D.

For example:

r=1, c=2 → sum 3 (in S), difference -1 (in D) → yes.

r=1, c=4 → sum 5 (in S), difference -3 (in D) → yes.

r=1, c=8 → sum 9 (in S), difference -7 → no.

So for r=1, c=2 and c=4 are valid.

For all (r,c):

We need to check if r + c is in S and r - c is in D.

This is O(36) operations.

Let's compute:

For each (r,c) in R x C:

r=1:

c=1: sum 2 (no), diff 0 → no.

c=2: sum 3 (yes), diff -1 (yes) → yes.

c=3: sum 4 (no), diff-2 → no.

c=4: sum5 (yes), diff-3 (yes) → yes.

c=5: sum6 (no), diff-4 → no.

c=8: sum9 (yes), diff-7 → no.

→ 2.

r=2:

c=1: sum3 (yes), diff1 (yes) → yes.

c=2: sum4 (no), diff0 → no.

c=3: sum5 (yes), diff-1 (yes) → yes.

c=4: sum6 (no), diff-2 → no.

c=5: sum7 (yes), diff-3 (yes) → yes.

c=8: sum10 (no), diff-6 → no.

→3.

r=3:

c=1: sum4 (no), diff2 → no.

c=2: sum5 (yes), diff1 (yes) → yes.

c=3: sum6 (no), diff0 → no.

c=4: sum7 (yes), diff-1 (yes) → yes.

c=5: sum8 (no), diff-2 → no.

c=8: sum11 (yes), diff-5 (yes) → yes.

→3.

r=4:

c=1: sum5 (yes), diff3 (yes) → yes.

c=2: sum6 (no), diff2 → no.

c=3: sum7 (yes), diff1 (yes) → yes.

c=4: sum8 (no), diff0 → no.

c=5: sum9 (yes), diff-1 (yes) → yes.

c=8: sum12 (no), diff-4 → no.

→3.

r=5:

c=1: sum6 (no), diff4 → no.

c=2: sum7 (yes), diff3 (yes) → yes.

c=3: sum8 (no), diff2 → no.

c=4: sum9 (yes), diff1 (yes) → yes.

c=5: sum10 (no), diff0 → no.

c=8: sum13 (no), diff-3 (yes) → no.

→2.

r=8:

c=1: sum9 (yes), diff7 → no.

c=2: sum10 (no), diff6 → no.

c=3: sum11 (yes), diff5 (yes) → yes.

c=4: sum12 (no), diff4 → no.

c=5: sum13 (no), diff3 (yes) → yes.

c=8: sum16 (no), diff0 → no.

→2.

Total ABCD: 2+3+3+3+2+2=15.

So subtract 15 → 56 -15=41.

So union_size = 152 -146 +50 -15 = 152-146=6 → 6+50=56 → 56-15=41.

So the answer is N^2 - union_size = 64 -41=23? But sample output is 2.

Wait, this indicates that the approach is incorrect. Because according to the sample input 1, the answer is 2. But according to this calculation, the answer is 64-41=23. Which is wrong.

So there's a mistake in the approach.

This suggests that the inclusion-exclusion approach is not correctly modeling the problem.

What's wrong here?

Let's think again.

The union_size computed via inclusion-exclusion is supposed to be the total number of squares that are in any of the rows, columns, sum diagonals, or difference diagonals of the existing pieces.

But according to the sample input 1, the answer is 2, which implies that the union_size is 64-2=62.

But according to the calculation above, union_size is 41, which is way off.

So there's a mistake in the approach.

Hmm. So where is the error?

Let me re-express the problem.

The correct answer for sample input 1 is 2. So the union_size must be 64-2=62.

But according to the inclusion-exclusion calculation, it's 41. So the approach is incorrect.

This suggests that the inclusion-exclusion model is not capturing all the squares that are under attack.

So what's wrong with the model?

The problem is that the existing pieces are not just their rows, columns, sum diagonals, and difference diagonals. Each existing piece's attack pattern includes all squares in their row, column, sum diagonal, and difference diagonal. So the union of all these lines is the set of squares under attack.

But the inclusion-exclusion model is supposed to capture exactly that.

But according to the sample input 1, the union_size should be 62. But according to the calculation, it's 41. So there's a discrepancy.

So the error must be in the calculation.

Let me re-calculate the individual terms for sample input 1.

Sample input 1:

R = {1,2,3,4,5,8} → 6 rows.

C = {1,2,3,4,5,8} → 6 columns.

S = {3,5,7,9,11} → 5 sum diagonals.

D = {-5, -3, -1, 1, 3,5} →6 difference diagonals.

A = 6 *8=48.

B =6*8=48.

C = sum for s in S:

s=3 →3-1=2.

s=5 →5-1=4.

s=7 →7-1=6.

s=9 →9-1=8.

s=11 →2*8 -11 +1= 16-11+1=6.

Total C=2+4+6+8+6=26.

D = sum for d in D:

d=-5 →8-5=3.

d=-3 →8-3=5.

d=-1 →8-1=7.

d=1 →8-1=7.

d=3 →8-3=5.

d=5 →8-5=3.

Total D=3+5+7+7+5+3=30.

Total individual:48+48+26+30=152.

Pairwise overlaps:

AB=6*6=36.

AC=21.

AD=22.

BC=21.

BD=22.

CD=24.

Total pairwise overlaps:36+21+22+21+22+24=146.

So 152-146=6.

Triple overlaps:50.

6+50=56.

Subtract ABCD=15 →56-15=41.

So union_size=41.

But sample input 1's answer is 2. So 64-41=23, which is not correct.

So the approach is wrong.

This suggests that the inclusion-exclusion model is incorrect.

What's the error?

Ah! The mistake is that the inclusion-exclusion model is not considering that each existing piece's square is counted in their own row, column, sum, and difference. But the existing pieces are already in the grid and are counted in the union. However, the problem requires that the new piece must be placed on an empty square, which is not occupied by any existing piece.

But according to the model, the union includes all existing pieces' squares. So the answer is N^2 - union_size.

But in sample input 1, the existing pieces are 6 squares, which are all part of the union. So the model's answer is 64-41=23, but the correct answer is 2. So the model is missing some squares.

But according to the sample explanation, the available squares are (6,6) and (7,7). So the union_size must be 64-2=62.

So the model is undercounting the union_size by 21.

So the inclusion-exclusion approach is not capturing all the squares that are under attack.

The problem is that the model is not considering that each existing piece's row, column, sum, and difference are all added to the union, but the inclusion-exclusion is not properly accounting for all overlaps.

Alternatively, perhaps the mistake is that the model is considering the sum of the individual line counts and their overlaps, but the lines can have overlaps beyond those considered in the pairwise, triple, and four-way overlaps.

But how?

Alternatively, perhaps the mistake is that the model is not considering that a single square can be covered by multiple lines from different pieces. For example, a square may be in the row of one piece and the column of another, but the model's inclusion-exclusion is not properly accounting for all such overlaps.

But according to the inclusion-exclusion formula, it should.

But why in the sample input 1, the model's calculation is incorrect.

Let me think of a small example.

Suppose N=2, M=1, piece at (1,1).

So R={1}, C={1}, S={2}, D={0}.

The union is all squares in row 1, column 1, sum 2, or difference 0.

Row 1: (1,1), (1,2).

Column 1: (1,1), (2,1).

Sum 2: (1,1).

Difference 0: (1,1), (2,2).

So union is:

(1,1), (1,2), (2,1), (2,2).

So union_size=4.

So answer is 2^2 -4=0.

But according to the model:

A=1*2=2.

B=1*2=2.

C= s=2 →2-1=1.

D= d=0 →2-0=2.

Total individual:2+2+1+2=7.

Pairwise overlaps:

AB=1*1=1.

AC= sum for r=1 and s=2: 1 <=2-1=1 <=2 → yes. So AC=1.

AD= r=1, d=0 → j=1-0=1 → valid. So AD=1.

BC= c=1, s=2 → i=2-1=1 → valid. So BC=1.

BD= c=1, d=0 → i=1+0=1 → valid. So BD=1.

CD= s=2, d=0 → (2+0)/2=1, (2-0)/2=1 → valid. So CD=1.

Pairwise overlaps total:1+1+1+1+1+1=6.

So 7-6=1.

Triple overlaps:

ABC= (1,1) in R x C, sum 2 → yes. So ABC=1.

ABD= (1,1) in R x C, difference 0 → yes. ABD=1.

ACD= r=1, s=2, d=0 → 2*1 -2=0 → d=0 is in D. j=2-1=1 → valid. So ACD=1.

BCD= c=1, s=2, d=0 → s-2c=0 → d=0 is in D. i=2-1=1 → valid. BCD=1.

Triple overlaps total:1+1+1+1=4.

So 1+4=5.

ABCD= (1,1) is in R x C, sum 2, difference 0 → yes. ABCD=1.

So union_size=5-1=4.

Which is correct. So the model works for this case.

But why is it not working for sample input 1?

Perhaps the error is in the calculation of the CD term.

In sample input 1, CD is 24.

But according to the sample input 1's CD calculation, the CD term is 24.

But perhaps the actual number of overlapping squares between sum and difference diagonals is more than 24.

Alternatively, perhaps the code for CD is incorrect.

Let me re-examine the CD calculation.

CD is the number of (s, d) pairs in S x D where (s +d) is even, and the resulting i and j are between 1 and N.

In code:

CD = sum(1 for s in S for d in D if (s + d) % 2 == 0 and ( (s + d) // 2 >= 1 ) and ( (s + d) // 2 <= N ) and ( (s - d) // 2 >= 1 ) and ( (s - d) // 2 <= N ) )

For sample input 1, S is {3,5,7,9,11}, D is {-5, -3, -1, 1, 3,5}.

For each s and d:

For example, s=3, d=-5:

s +d =-2 → even.

i = (-2)/2 =-1 → invalid.

So not counted.

s=3, d=-3 → sum 0 → even.

i=0 → invalid.

s=3, d=-1 → sum 2 → even.

i=1, j=2 → valid.

So counted.

s=3, d=1 → sum 4 → even.

i=2, j=1 → valid.

s=3, d=3 → sum 6 → even.

i=3, j=0 → invalid.

s=3, d=5 → sum 8 → even.

i=4, j= (3-5)/2 =-1 → invalid.

So for s=3, 2 valid.

Similarly, for s=5, d=-5:

sum 0 → even.

i=0 → invalid.

d=-3 → sum 2 → even.

i=1, j=4 → valid.

d=-1 → sum 4 → even.

i=2, j=3 → valid.

d=1 → sum 6 → even.

i=3, j=2 → valid.

d=3 → sum 8 → even.

i=4, j=1 → valid.

d=5 → sum 10 → even.

i=5, j=0 → invalid.

So 4 valid.

For s=5, 4 valid.

For s=7, d=-5:

sum 2 → even.

i=1, j=6 → valid.

d=-3 → sum 4 → even.

i=2, j=5 → valid.

d=-1 → sum 6 → even.

i=3, j=4 → valid.

d=1 → sum 8 → even.

i=4, j=3 → valid.

d=3 → sum 10 → even.

i=5, j=2 → valid.

d=5 → sum 12 → even.

i=6, j=1 → valid.

So 6 valid.

s=7, d=-5: yes.

s=9:

d=-5 → sum 4 → even.

i=2, j=7 → valid.

d=-3 → sum 6 → even.

i=3, j=6 → valid.

d=-1 → sum 8 → even.

i=4, j=5 → valid.

d=1 → sum 10 → even.

i=5, j=4 → valid.

d=3 → sum 12 → even.

i=6, j=3 → valid.

d=5 → sum 14 → even.

i=7, j=2 → valid.

So 6 valid.

s=11:

d=-5 → sum 6 → even.

i=3, j=8 → valid.

d=-3 → sum 8 → even.

i=4, j=7 → valid.

d=-1 → sum 10 → even.

i=5, j=6 → valid.

d=1 → sum 12 → even.

i=6, j=5 → valid.

d=3 → sum 14 → even.

i=7, j=4 → valid.

d=5 → sum 16 → even.

i=8, j=3 → valid.

So 6 valid.

So CD is 2+4+6+6+6=24, which is correct.

So why the model's calculation for sample input 1 is incorrect.

Another possibility is that the model is not considering that the existing pieces are part of the union. But according to the model, the existing pieces are included in the rows, columns, sum diagonals, and difference diagonals. So their squares are counted in the union.

But according to the sample input 1, the union_size should be 62, but the model's calculation gives 41.

So there's a discrepancy of 21.

What is the correct union_size for sample input 1?

The sample input 1's output is 2, so the union_size must be 64-2=62.

So the model's calculation is missing 62-41=21 squares.

So where are these 21 squares?

The model's calculation is missing some squares that are under attack.

So the inclusion-exclusion approach is not capturing all the squares that are under attack.

The error must be in the model.

Alternative approach: perhaps the mistake is that the model is not considering that each existing piece's row, column, sum, and difference are all added to the union, but the way the overlaps are computed is incorrect.

Alternatively, perhaps the mistake is that the model is not considering that a square can be covered by multiple lines from the same piece.

For example, a piece's row and column are considered, but the square is counted once.

But the inclusion-exclusion model should handle that.

Alternatively, perhaps the model is not considering that a square can be covered by multiple pieces' lines.

But the inclusion-exclusion model should handle that as well.

But why for sample input 1, the model's calculation is incorrect.

Let me think of the actual squares under attack in sample input 1.

The existing pieces are at:

(1,4), (2,1), (3,8), (4,5), (5,2), (8,3).

Each of these pieces attacks their row, column, sum diagonal, and difference diagonal.

So let's compute the union of all these lines.

For example, the row 1 has 8 squares.

Column 4 has 8 squares.

Sum diagonal 1+4=5: squares (1,4), (2,3), (3,2), (4,1).

Difference diagonal 1-4=-3: squares (1,4), (2,5), (3,6), (4,7), (5,8).

But the piece (1,4) is already counted in the row, column, sum, and difference.

So the union of all lines from all pieces includes:

- All rows of the existing pieces: rows 1,2,3,4,5,8.

- All columns of existing pieces: columns 1,2,3,4,5,8.

- All sum diagonals of existing pieces: 5,3,11,9,7.

- All difference diagonals of existing pieces: -3,1,-5,-1,3,5.

So the union is the set of squares that are in any of these rows, columns, sum diagonals, or difference diagonals.

The total number of squares in the union is 62.

So the model's calculation of 41 is missing 21 squares.

So the inclusion-exclusion approach is not working.

Alternative idea: perhaps the model is not considering that the same square can be covered by multiple lines from different pieces, and the inclusion-exclusion formula is not properly accounting for all overlaps.

But how?

Alternatively, perhaps the mistake is that the model is not considering that the lines from different pieces can overlap in ways that are not captured by the current pairwise, triple, and four-way overlaps.

But this is unclear.

Alternative approach: think of the problem in terms of the four sets:

Rows: all squares in any of the R rows.

Columns: all squares in any of the C columns.

Sum diagonals: all squares in any of the S sum diagonals.

Difference diagonals: all squares in any of the D difference diagonals.

The union is the set of squares that are in Rows OR Columns OR Sum OR Difference.

The size of the union can be computed as:

size(Rows) + size(Columns) + size(Sum) + size(Difference) 

- size(Rows ∩ Columns) - size(Rows ∩ Sum) - size(Rows ∩ Difference) - size(Columns ∩ Sum) - size(Columns ∩ Difference) - size(Sum ∩ Difference) 

+ size(Rows ∩ Columns ∩ Sum) + size(Rows ∩ Columns ∩ Difference) + size(Rows ∩ Sum ∩ Difference) + size(Columns ∩ Sum ∩ Difference) 

- size(Rows ∩ Columns ∩ Sum ∩ Difference)

But according to the model, this is what is being computed.

But in sample input 1, this formula gives 41, but the correct union size is 62.

So the model is incorrect.

But why?

Alternative idea: perhaps the calculation of the individual terms is incorrect.

For example, the sum of the rows is len(R) * N =6 *8=48.

But the actual number of squares in the rows is 6*8=48.

Similarly for columns: 6*8=48.

Sum diagonals: 26.

Difference diagonals:30.

Total individual:48+48+26+30=152.

But the actual union is 62.

So the model's calculation of the individual terms is correct, but the overlapping terms are not.

But the overlapping terms are subtracted, but perhaps the overlapping terms are not computed correctly.

For example, AB is the number of squares in both R rows and C columns, which is 6*6=36. But each such square is (r,c) where r is in R and c is in C. But each of these squares is a single square. So 36 squares.

But in reality, the rows and columns may overlap, but the actual number of squares in the intersection of rows and columns is 36.

But in the sample input 1, R and C are both 6 elements. So the intersection of rows and columns is 6*6=36 squares.

But according to the sample input 1's union size of 62, these 36 squares are part of the union.

But according to the model's calculation, the union_size is 41, which is less than the actual 62.

So the model is not capturing all the squares.

This suggests that the model's approach is fundamentally incorrect.

Alternative idea: perhaps the mistake is that the model is not considering that the lines (rows, columns, sum diagonals, difference diagonals) are not independent. For example, a row can be part of a sum diagonal, but the model's inclusion-exclusion is not capturing this.

But how?

Alternatively, perhaps the mistake is that the model is not considering that the same square can be covered by multiple lines from the same piece.

But the inclusion-exclusion should handle this.

But perhaps the model is not considering that the lines from the same piece can overlap.

For example, a piece's row and column intersect at the piece's square, which is counted once.

But the model's inclusion-exclusion is designed to handle overlaps between different lines, regardless of which piece they come from.

But the problem is that the lines are from all pieces, and the model's approach is to consider all possible lines from all pieces.

But perhaps the model's approach is correct, but the sample input 1's calculation is incorrect due to a miscalculation in the code.

Alternatively, perhaps the mistake is in the way the code calculates the sum of the sum diagonals and difference diagonals.

Wait, let's re-calculate the sum of the sum diagonals for sample input 1.

S = {3,5,7,9,11}

For each s in S:

s=3: squares (1,2), (2,1) → 2 squares.

s=5: (1,4), (2,3), (3,2), (4,1) →4 squares.

s=7: (1,6), (2,5), (3,4), (4,3), (5,2), (6,1) →6 squares.

s=9: (1,8), (2,7), (3,6), (4,5), (5,4), (6,3), (7,2), (8,1) →8 squares.

s=11: (3,8), (4,7), (5,6), (6,5), (7,4), (8,3) →6 squares.

Total sum diagonals: 2+4+6+8+6=26. Correct.

Difference diagonals D = {-5, -3, -1, 1, 3,5}.

For each d in D:

d=-5: squares (1,6), (2,7), (3,8) →3 squares.

d=-3: (1,4), (2,5), (3,6), (4,7), (5,8) →5 squares.

d=-1: (1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,8) →7 squares.

d=1: (2,1), (3,2), (4,3), (5,4), (6,5), (7,6), (8,7) →7 squares.

d=3: (4,1), (5,2), (6,3), (7,4), (8,5) →5 squares.

d=5: (6,1), (7,2), (8,3) →3 squares.

Total difference diagonals: 3+5+7+7+5+3=30. Correct.

So the individual terms are correct.

The problem must be in the overlapping terms.

But according to the model's calculation, the union_size is 41, but the correct answer is 62.

So the model is missing 21 squares.

What's the source of this discrepancy?

Alternative idea: perhaps the model is not considering that the same square can be covered by multiple lines from different pieces, but the inclusion-exclusion is not properly accounting for this.

But according to the inclusion-exclusion formula, it should.

But why in the sample input 1, the model's calculation is incorrect.

Alternative approach: let's compute the union_size manually for sample input 1.

The union includes:

- All squares in rows 1, 2, 3, 4, 5, 8 → 6 rows ×8=48 squares.

- All squares in columns 1, 2, 3, 4, 5, 8 → 6 columns ×8=48 squares.

- All squares in sum diagonals 3,5,7,9,11 →26 squares.

- All squares in difference diagonals -5, -3, -1, 1, 3,5 →30 squares.

The total is 48+48+26+30=152. But overlapping squares are counted multiple times.

The actual union size is 62.

So the overlapping squares are 152-62=90.

But according to the model's calculation, the overlapping squares are 146-50+15=101, which is 152-41=111, which is not matching.

So the model's calculation of overlapping squares is incorrect.

This suggests that the inclusion-exclusion approach is not correctly modeling the overlaps.

But why?

Perhaps the mistake is that the model is not considering that a square can be in multiple lines from different categories. For example, a square can be in a row from one piece, a column from another piece, and a sum diagonal from a third piece.

But the inclusion-exclusion formula should account for all possible overlaps.

But according to the sample input 1's calculation, the model is missing 21 squares.

This suggests that the model is not capturing all the overlaps.

Alternatively, perhaps the mistake is that the model is not considering that some squares are covered by multiple lines of the same type. For example, a square may be in multiple rows, but the model's calculation of rows is len(R) * N, which assumes that all rows in R are unique. Which they are.

But R is a set, so rows are unique.

So the model's calculation of rows is correct.

But the problem is that the model is not considering that a square can be in a row from one piece and a column from another piece, which is captured in the AB term.

But according to the model's calculation for sample input 1, AB is 36, which is the number of squares in rows and columns.

But the actual number of squares in rows and columns is 6*8 + 6*8 - 36 = 48+48-36=60. Which is the total of rows and columns minus their intersections.

But the union of rows and columns is 60 squares.

But according to the model's calculation, the union of rows and columns is 48+48-36=60, which is correct.

But the model's calculation for the union of rows, columns, sum diagonals, and difference diagonals is 41, but the actual union is 62.

So the model is missing 62-41=21 squares.

But where are these squares?

The sum diagonals and difference diagonals contribute 26+30=56 squares, but overlapping with rows and columns.

But according to the model's calculation, the sum of the individual terms (152) minus the pairwise overlaps (146) plus the triple overlaps (50) minus the four-way overlaps (15) gives 41.

But the actual union is 62.

So the model's calculation is incorrect.

This suggests that the inclusion-exclusion approach is not suitable for this problem, possibly due to higher-order overlaps that are not being properly considered.

Alternative idea: perhaps the problem can be modeled by considering that a square is under attack if it is in any of the four types of lines of any of the existing pieces. So the square is under attack if:

i ∈ R OR j ∈ C OR (i + j) ∈ S OR (i - j) ∈ D.

The number of squares under attack is the number of (i,j) in [1..N]^2 that satisfy this condition.

This can be computed as:

number of squares where i ∈ R OR j ∈ C OR (i+j) ∈ S OR (i-j) ∈ D.

The problem is to compute the number of such squares.

This is equivalent to the union of the four sets.

But how to compute this efficiently.

The key insight is that the four conditions are independent, but overlapping. To compute the size of the union, we can use the inclusion-exclusion principle for four sets.

But the inclusion-exclusion formula for four sets is:

|A ∪ B ∪ C ∪ D| = |A| + |B| + |C| + |D| 

- |A∩B| - |A∩C| - |A∩D| - |B∩C| - |B∩D| - |C∩D| 

+ |A∩B∩C| + |A∩B∩D| + |A∩C∩D| + |B∩C∩D| 

- |A∩B∩C∩D| 

In this case:

A is the set of squares in any of the rows R.

B is the set of squares in any of the columns C.

C is the set of squares in any of the sum diagonals S.

D is the set of squares in any of the difference diagonals D.

So the model's approach is correct.

But according to the sample input 1, the model's calculation gives 41, but the correct answer is 62.

So the error must be in the calculation of the terms.

But according to the previous calculation, the model's terms are correct.

So this suggests that there's a mistake in the code logic.

But why?

Alternatively, perhaps the mistake is in the way the code calculates the sum of the sum diagonals and difference diagonals.

Wait, for sum diagonals, the code computes for each s in S: (s-1) if s <= N+1 else (2N - s +1).

But for N=8 and s=11: 2*8 -11 +1= 16-11+1=6. Which is correct.

Yes.

For difference diagonals, the code computes N - abs(d) for each d in D.

For d=5: 8-5=3. Correct.

Yes.

So the individual terms are correct.

The problem must be in the overlapping terms.

But according to the model's calculation, the overlapping terms sum to 146, which is subtracted from 152, giving 6. Then adding 50 and subtracting 15 gives 41.

But the actual union_size is 62.

So the model is missing 62-41=21 squares.

So where are these 21 squares?

Alternative idea: perhaps the model is not considering that some squares are covered by multiple lines of the same type. For example, a square may be in multiple sum diagonals from different pieces.

But the model's calculation of sum_sum is the sum of the number of squares in each sum diagonal in S, but if two sum diagonals overlap, their overlapping squares are counted multiple times. However, the inclusion-exclusion model is supposed to subtract the overlapping counts.

But the sum_sum is the sum of the sizes of the sum diagonals, which includes overlaps between sum diagonals. But the inclusion-exclusion model only subtracts overlaps between sum diagonals and other types of lines, not between sum diagonals themselves.

Ah! This is the mistake.

The model's approach treats the sum diagonals as a single group, but the sum diagonals may overlap with each other. Similarly for the difference diagonals.

The inclusion-exclusion model is only considering overlaps between different types of lines (rows, columns, sum, difference), but not overlaps within the same type.

For example, two different sum diagonals may overlap, and the model's current approach counts both, but in reality, the overlapping squares are counted multiple times in the individual sum_sum term.

But the inclusion-exclusion formula is designed to handle overlaps between different types of sets, but not within the same type.

This is the critical error.

The model's approach treats the sum of the individual terms as the sum of all rows, columns, sum diagonals, and difference diagonals, but this sum includes overlaps within the same type (e.g., two sum diagonals overlapping). The inclusion-exclusion formula subtracts overlaps between different types, but not within the same type.

Thus, the model's approach overcounts the individual terms and undercounts the overlaps, leading to an incorrect union_size.

This explains why the sample input 1's model calculation is incorrect.

The correct approach must account for overlaps within the same type of lines (e.g., sum diagonals overlapping with other sum diagonals).

But this makes the problem much more complex, as it requires considering all possible overlaps between all lines, including those of the same type.

But with M up to 1e3, this is computationally infeasible.

Thus, the inclusion-exclusion approach is not suitable for this problem.

Alternative approach:

The problem requires counting the number of squares (i,j) that are not in any of the rows of existing pieces, not in any of the columns of existing pieces, not in any of the sum diagonals of existing pieces, and not in any of the difference diagonals of existing pieces.

The number of such squares is equal to the total number of squares minus the number of squares that are in at least one of the rows, columns, sum diagonals, or difference diagonals.

The key insight is to compute the number of squares that are in any of the rows, columns, sum diagonals, or difference diagonals.

But how to compute this efficiently.

Alternative idea: compute the number of squares not in any of the rows, columns, sum diagonals, or difference diagonals.

The number of valid squares is:

(N - |R|) * (N - |C|) - [squares not in R or C but in sum or difference diagonals]

But this is not straightforward.

Alternative approach: use the principle of inclusion-exclusion, but consider all four types of lines, including overlaps within the same type.

But this would require considering all possible overlaps between all lines of all types, which is computationally infeasible for M=1e3.

Thus, the problem requires a different approach.

Alternative idea: for a square (i,j) to be valid, it must satisfy:

i ∉ R,

j ∉ C,

i + j ∉ S,

i - j ∉ D.

The number of such squares is equal to the number of squares that are not in R, not in C, not in S, and not in D.

But how to compute this.

The number of squares not in R is N - |R|.

The number of squares not in C is N - |C|.

The number of squares not in S is N^2 - sum of squares in S.

But this is not correct, because the squares not in S are those not in any of the sum diagonals of existing pieces.

But this is not the same as (i + j) not in S.

Thus, the valid squares are those that are not in any row of existing pieces, not in any column of existing pieces, not in any sum diagonal of existing pieces, and not in any difference diagonal of existing pieces.

This is equivalent to:

i ∉ R AND j ∉ C AND (i + j) ∉ S AND (i - j) ∉ D.

The number of such squares is the answer.

But how to compute this.

The key is to compute the number of squares (i,j) that satisfy all four conditions.

This can be computed as:

Total squares not in R or C or S or D = N^2 - |R ∪ C ∪ S ∪ D|.

But this is the same as the initial approach.

Thus, the problem reduces to computing the size of the union of the four sets.

But the initial approach's inclusion-exclusion model is incorrect because it does not account for overlaps within the same type of lines.

Thus, the correct approach must account for all overlaps, including those between lines of the same type.

But this is computationally infeasible for large M.

Alternative idea: the problem can be transformed into a mathematical formula that considers the four conditions.

The number of valid squares is:

(N - |R|) * (N - |C|) - [number of squares not in R and not in C but in S or D].

But how to compute this.

The first term (N - |R|) * (N - |C|) is the number of squares not in any row of R and not in any column of C.

From these squares, we need to subtract those that are in any sum diagonal of S or any difference diagonal of D.

So the valid squares are:

valid = (N - |R|) * (N - |C|) - (A + B - C),

where:

A is the number of squares not in R and not in C but in S.

B is the number of squares not in R and not in C but in D.

C is the number of squares not in R and not in C but in both S and D.

Thus, the formula is:

valid = (N - |R|) * (N - |C|) - (A + B - C)

But how to compute A, B, and C.

A is the number of squares (i,j) such that i ∉ R, j ∉ C, and (i + j) ∈ S.

B is the number of squares (i,j) such that i ∉ R, j ∉ C, and (i - j) ∈ D.

C is the number of squares (i,j) such that i ∉ R, j ∉ C, (i + j) ∈ S, and (i - j) ∈ D.

Thus, the problem reduces to computing A, B, and C.

But how to compute these terms efficiently.

Let's compute each term:

A: For each s in S, compute the number of squares (i,j) in the sum diagonal s, but i ∉ R and j ∉ C.

For a sum diagonal s, the squares are those where i + j = s, i >=1, j >=1, i <=N, j <=N.

The number of such squares is the length of the diagonal, which is:

len = s-1 if s <= N+1 else 2*N - s +1.

But we need to subtract the squares where i is in R or j is in C.

But this is the inclusion-exclusion for each diagonal s:

number of valid squares for s is len - (number of squares in s that are in R or C) + (number of squares in s that are in R and C).

But this seems complicated.

Alternative approach: for each s in S, compute the number of (i,j) in the sum diagonal s where i ∉ R and j ∉ C.

For each s in S:

The sum diagonal s has i + j = s.

For each i in [max(1, s - N), min(N, s - 1)], j = s - i.

We need to count the number of (i,j) where i ∉ R and j ∉ C.

This can be computed as:

count = 0

for each i in the valid range for s:

j = s - i

if i not in R and j not in C:

count +=1

But for large N, this is not feasible.

But M is up to 1e3, and S is up to 1e3 elements.

For each s in S, the valid i range is up to min(N, s-1) - max(1, s-N) +1, which is up to N steps.

But for N up to 1e9, this is not feasible.

Thus, we need a mathematical way to compute this.

For a given s, the valid squares (i,j) in the sum diagonal s are those where i is not in R and j = s - i is not in C.

We can model this as:

i ranges from max(1, s - N) to min(N, s -1).

For each i in this range, j = s -i.

We need i ∉ R and j ∉ C.

The number of such i is equal to the number of integers in the i range that are not in R, and for which j = s -i is not in C.

This can be computed as follows:

The total number of i in the range is len = min(N, s-1) - max(1, s - N) + 1.

The number of i in the range that are in R is the number of R elements in the i range.

The number of j in the range that are in C is the number of C elements in the j range.

But j = s -i, so for each i in the range, j = s -i.

The j values are s - i, which ranges from s - min(N, s-1) to s - max(1, s - N).

Which is:

s - min(N, s-1) = max(1, s - N) → j ranges from max(1, s - N) to min(N, s-1).

Wait, no. For i in [a, b], j = s -i is in [s -b, s -a].

So the j range is [s - b, s -a].

But this is the same as the i range, but reversed.

Thus, for each s, the j values are the same as the i values, but in reverse.

Thus, the number of j in C is the number of elements in C that are in the j range.

But how to compute this.

The problem is to find, for each s in S, the number of i in the i range such that i ∉ R and (s -i) ∉ C.

This is equivalent to the total number of i in the range minus the number of i in R or (s -i) in C.

But using inclusion-exclusion:

count = total_i - (num_i_in_R + num_j_in_C - num_i_in_R_and_j_in_C)

where:

total_i is the length of the i range.

num_i_in_R is the number of i in the i range that are in R.

num_j_in_C is the number of j in the j range (which is s -i for i in the i range) that are in C.

num_i_in_R_and_j_in_C is the number of i in the i range that are in R and j = s -i is in C.

But how to compute these terms.

For each s in S:

Compute the i range [a, b], where a = max(1, s - N), b = min(N, s-1).

The j range is [s - b, s - a].

But j = s -i, so j ranges from s -b to s -a.

But since i ranges from a to b, j ranges from s -b to s -a, which is the same as j from a to b (since s -b = a and s -a = b when i ranges from a to b and j = s -i).

Wait, no. For example, if s=5 and N=8:

i ranges from max(1,5-8)=1 to min(8,5-1)=4.

j ranges from 5-4=1 to 5-1=4.

So j ranges from 1 to4, which is the same as i's range.

Thus, for each s, the j range is the same as the i range.

Thus, the j range is [a, b], where a = max(1, s - N), b = min(N, s-1).

So for each s in S:

num_i_in_R = number of elements in R that are in [a, b].

num_j_in_C = number of elements in C that are in [a, b].

num_i_in_R_and_j_in_C = number of i in R ∩ [a, b] such that (s -i) is in C.

Thus, the count for s is:

count = (b - a + 1) - (num_i_in_R + num_j_in_C - num_i_in_R_and_j_in_C)

This can be computed for each s in S.

But how to compute these terms efficiently.

For each s in S:

Compute a and b.

num_i_in_R can be computed by checking the R set for elements in [a, b].

But R is a set, so for each s, iterate through R and count how many are between a and b.

But with R up to 1e3, this is manageable.

Similarly for C.

num_i_in_R_and_j_in_C can be computed by iterating through each i in R that is in [a, b], and checking if (s -i) is in C.

This is O(M) per s, which is manageable for M=1e3.

Thus, for each s in S, the computation is O(M) time.

Similarly for D.

Thus, the total time complexity would be O(M^2), which is acceptable.

Similarly for B and C.

Thus, the approach is:

Compute the number of valid squares as:

valid = (N - |R|) * (N - |C|) - (A + B - C)

where:

A is the sum over s in S of the number of squares in sum diagonal s that are not in R or C.

B is the sum over d in D of the number of squares in difference diagonal d that are not in R or C.

C is the number of squares that are in both a sum diagonal in S and a difference diagonal in D, and are not in R or C.

Thus, the steps are:

1. Compute R, C, S, D.

2. Compute the initial term: (N - len(R)) * (N - len(C)).

3. Compute A: for each s in S, compute the number of squares in sum diagonal s that are not in R or C.

4. Compute B: for each d in D, compute the number of squares in difference diagonal d that are not in R or C.

5. Compute C: the number of squares that are in both a sum diagonal in S and a difference diagonal in D, and are not in R or C.

6. valid = initial - (A + B - C)

The answer is valid.

Now, let's compute this for sample input 1.

Sample input 1:

R = {1,2,3,4,5,8}

C = {1,2,3,4,5,8}

S = {3,5,7,9,11}

D = {-5, -3, -1, 1, 3,5}

N=8.

Initial term: (8-6)*(8-6) = 2*2=4.

Compute A:

For each s in S:

s=3:

a = max(1, 3-8)=1, b= min(8,3-1)=2.

i ranges from 1 to 2.

For each i in [1,2], j=3-i.

Check i not in R and j not in C.

R includes 1,2, so i=1 and 2 are in R. So no squares in s=3 contribute to A.

A for s=3 is 0.

s=5:

a= max(1,5-8)=1, b= min(8,4)=4.

i ranges 1-4.

For i in R: 1,2,3,4.

So for i not in R: none.

Thus, A for s=5 is 0.

s=7:

a= max(1,7-8)=1, b= min(8,6)=6.

i ranges 1-6.

i not in R: 6 is not in R (R is {1,2,3,4,5,8}).

So i=6.

j=7-6=1, which is in C. So not counted.

Thus, A for s=7 is 0.

s=9:

a= max(1,9-8)=1, b= min(8,8)=8.

i ranges 1-8.

i not in R: 6,7.

For i=6: j=9-6=3 → in C. Not counted.

For i=7: j=9-7=2 → in C. Not counted.

Thus, A for s=9 is 0.

s=11:

a= max(1,11-8)=3, b= min(8,10)=8.

i ranges 3-8.

i not in R: 6,7.

For i=6: j=11-6=5 → in C. Not counted.

For i=7: j=11-7=4 → in C. Not counted.

Thus, A for s=11 is 0.

Total A=0.

Compute B:

For each d in D:

d=-5:

difference diagonal: i -j =-5 → j =i +5.

i ranges from 1 to 3 (since j =i+5 <=8 → i <=3).

For i in 1-3:

i not in R: i=6,7, etc. But i ranges 1-3, all in R. So no squares contribute.

B for d=-5: 0.

d=-3:

i -j =-3 → j =i +3.

i ranges 1-5 (j <=8).

i in R: 1,2,3,4,5.

No i not in R. So B for d=-3: 0.

d=-1:

j =i +1.

i ranges 1-7.

i not in R: 6,7.

For i=6: j=7 → not in C (C is {1,2,3,4,5,8}).

So counted.

For i=7: j=8 → in C. Not counted.

Thus, B for d=-1: 1.

d=1:

j =i -1.

i ranges 2-8.

i not in R:6,7.

For i=6: j=5 → in C. Not counted.

For i=7: j=6 → not in C. Counted.

Thus, B for d=1: 1.

d=3:

j =i -3.

i ranges 4-8.

i not in R:6,7.

For i=6: j=3 → in C. Not counted.

For i=7: j=4 → in C. Not counted.

Thus, B for d=3: 0.

d=5:

j =i -5.

i ranges 6-8.

i not in R:6,7.

For i=6: j=1 → in C. Not counted.

For i=7: j=2 → in C. Not counted.

Thus, B for d=5:0.

Total B=1+1=2.

Compute C:

C is the number of squares (i,j) not in R or C, and in S and D.

For each s in S and d in D:

Check if (s +d) is even, and i=(s +d)/2, j=(s -d)/2 are integers between 1 and 8.

Also, i not in R, j not in C.

For example:

s=3, d=-1:

s +d =2 → even.

i=1, j=2.

i=1 is in R. So not counted.

s=5, d=1:

s +d=6 → even.

i=3, j=2.

i=3 is in R. Not counted.

s=7, d=1:

s +d=8 → even.

i=4, j=3.

i=4 is in R. Not counted.

s=7, d=-1:

s +d=6 → even.

i=3, j=4.

i=3 is in R. Not counted.

s=7, d=3:

s +d=10 → even.

i=5, j=2.

i=5 is in R. Not counted.

s=9, d=-5:

s +d=4 → even.

i=2, j=7.

i=2 is in R. Not counted.

s=9, d=-3:

s +d=6 → even.

i=3, j=6.

i=3 is in R. Not counted.

s=9, d=1:

s +d=10 → even.

i=5, j=4.

i=5 is in R. Not counted.

s=9, d=3:

s +d=12 → even.

i=6, j=3.

i=6 not in R, j=3 is in C. So not counted.

s=11, d=-5:

s +d=6 → even.

i=3, j=8.

i=3 is in R. Not counted.

s=11, d=-3:

s +d=8 → even.

i=4, j=7.

i=4 is in R. Not counted.

s=11, d=1:

s +d=12 → even.

i=6, j=5.

i=6 not in R, j=5 is in C. Not counted.

s=11, d=3:

s +d=14 → even.

i=7, j=4.

i=7 not in R, j=4 is in C. Not counted.

s=5, d=-5:

s +d=0 → even.

i=0 → invalid.

s=5, d=5:

s +d=10 → even.

i=5, j=0 → invalid.

Thus, after checking all (s,d) pairs, there are no squares that are in both S and D, not in R or C.

Thus, C=0.

Thus, valid =4 - (0 +2 -0) =4-2=2.

Which matches the sample input 1's output.

Thus, this approach is correct.

Thus, the correct way to model the problem is to compute the initial term (N - |R|) * (N - |C|) and subtract the squares that are in sum or difference diagonals but not in R or C, considering their overlaps.

Thus, the code should be written according to this approach.

Now, the task is to implement this approach in Python.

The steps are:

1. Read N and M.

2. Read M pairs (a, b), collect R, C, S, D as sets.

3. Compute initial = (N - len(R)) * (N - len(C)).

4. Compute A: for each s in S, compute the number of squares in sum diagonal s that are not in R or C.

5. Compute B: for each d in D, compute the number of squares in difference diagonal d that are not in R or C.

6. Compute C: the number of squares that are in both S and D, and not in R or C.

7. valid = initial - (A + B - C).

8. Output valid.

Now, implementing this.

For step 4:

For each s in S:

a = max(1, s - N)

b = min(N, s-1)

i ranges from a to b.

The number of i in [a, b] not in R and j = s - i not in C.

This can be computed as follows:

count = 0

for i in a to b:

if i not in R and (s - i) not in C:

count +=1

But for large N, this is not feasible.

But M is up to 1e3, and S is up to 1e3, and for each s in S, the range [a, b] can be up to N, but N is up to 1e9.

Thus, we need a mathematical way to compute this.

The number of i in [a, b] not in R and (s -i) not in C is equal to:

(total_i) - (num_i_in_R + num_j_in_C - num_i_in_R_and_j_in_C)

where:

total_i = b - a + 1

num_i_in_R is the number of i in R that are in [a, b]

num_j_in_C is the number of j = s -i in C, where i is in [a, b]

num_i_in_R_and_j_in_C is the number of i in R in [a, b] such that s -i is in C.

But how to compute these terms.

num_i_in_R is the number of elements in R that are between a and b inclusive.

num_j_in_C is the number of elements in C that are between (s - b) and (s - a) inclusive.

But since j = s -i, and i ranges from a to b, j ranges from s -b to s -a.

But s -b = a (since i ranges from a to b, and j = s -i → j ranges from s -b to s -a).

Wait, i ranges from a to b:

i = a → j = s -a

i = b → j = s -b

But a <= b → s -b <= s -a.

But since a and b are defined as a = max(1, s - N) and b = min(N, s-1), j ranges from s -b to s -a.

But for example, s=5, N=8:

a=1, b=4.

j ranges from 5-4=1 to 5-1=4.

So j ranges from 1 to4.

Thus, j ranges from a to b.

Thus, the j range is [a, b].

Thus, num_j_in_C is the number of elements in C that are between a and b.

num_i_in_R_and_j_in_C is the number of i in R ∩ [a, b] such that (s -i) is in C.

This can be computed by looping through each i in R and checking if i is in [a, b] and (s -i) is in C.

Thus, for each s in S:

compute a and b.

num_i_in_R = number of elements in R that are between a and b.

num_j_in_C = number of elements in C that are between a and b.

num_i_in_R_and_j_in_C = sum(1 for i in R if a <=i <=b and (s -i) in C)

Thus, the count for s is:

count = (b - a +1) - (num_i_in_R + num_j_in_C - num_i_in_R_and_j_in_C)

Thus, this can be computed efficiently.

Similarly for B.

For B:

For each d in D:

The difference diagonal is i - j =d → j =i -d.

i ranges from max(1, d +1) to min(N, N +d).

But j must be between 1 and N.

Thus, i ranges from max(1, d +1) to min(N, N +d).

But j =i -d >=1 → i >=d +1.

j =i -d <=N → i <=N +d.

But i must also be <=N.

Thus, the valid i range is [max(1, d+1), min(N, N +d)].

But this can be rewritten as:

a = max(1, d +1)

b = min(N, N +d)

But this may be empty.

For example, d=5:

a =5+1=6, b=8 +5=13 → min(8,13)=8.

Thus, i ranges 6-8.

j =i -5 ranges 1-3.

Thus, for each d in D:

compute a = max(1, d +1)

b = min(N, N +d)

if a >b: continue.

The number of squares in this difference diagonal is b -a +1.

But we need to compute the number of squares (i,j) in this diagonal where i not in R and j not in C.

This is similar to the sum diagonal case.

The number of such squares is:

count = (b -a +1) - (num_i_in_R + num_j_in_C - num_i_in_R_and_j_in_C)

where:

num_i_in_R is the number of i in R ∩ [a, b]

num_j_in_C is the number of j in C ∩ [a -d, b -d] → since j =i -d, i ranges [a, b] → j ranges [a -d, b -d]

Wait, j =i -d.

i ranges from a to b.

j ranges from a -d to b -d.

But a = max(1, d +1) → a -d = max(1 -d, 1).

But this may not be the same as the j range.

For example, d=5, a=6, b=8:

j ranges from 6-5=1 to 8-5=3.

Thus, j ranges from 1 to3.

Thus, j ranges [1,3].

Thus, num_j_in_C is the number of elements in C between 1 and3.

Thus, for difference diagonal d:

j ranges from (a -d) to (b -d).

But a = max(1, d +1) → a -d = max(1 -d, 1).

But this may be complicated.

But since j =i -d and i ranges [a, b], j ranges [a -d, b -d].

But a and b are defined such that j >=1 and i <=N.

But perhaps it's easier to compute j range as [a -d, b -d], and then clamp it to [1, N].

But for the purposes of counting num_j_in_C, we can compute the actual j range.

But for the difference diagonal d:

j =i -d.

i ranges [a, b] → j ranges [a -d, b -d].

But since i >=a = max(1, d +1), j >= a -d = max(1, d+1) -d = max(1 -d, 1).

But this can be negative.

For example, d=-5:

a = max(1, -5+1)=1.

b= min(8, 8 + (-5))=3.

j ranges from 1 - (-5) =6 to 3 - (-5)=8.

Thus, j ranges 6-8.

But j must be <=N=8.

Thus, j ranges 6-8.

Thus, for difference diagonal d=-5, the j range is [6,8].

Thus, num_j_in_C is the number of elements in C between 6 and8.

But C is {1,2,3,4,5,8}.

So elements in C between 6 and8: 8.

Thus, num_j_in_C=1.

Thus, for difference diagonal d=-5:

a=1, b=3.

i ranges 1-3.

num_i_in_R = number of R elements in [1,3] → R has 1,2,3 → 3.

num_j_in_C =1.

num_i_in_R_and_j_in_C = number of i in R ∩ [1,3] where j =i -d =i +5 is in C.

d=-5 → j= i +5.

i ranges 1-3:

i=1 → j=6 → not in C.

i=2 → j=7 → not in C.

i=3 → j=8 → in C.

Thus, num_i_in_R_and_j_in_C=1.

Thus, count = (3 -1 +1) - (3 +1 -1) =3 -3=0.

Thus, for d=-5, B contribution is 0.

Thus, the code must compute this correctly.

Thus, the code for B is similar to A, but with the difference diagonal's i range and j calculation.

Finally, C is the number of squares (i,j) not in R or C, and in S and D.

This can be computed by iterating through all s in S and d in D, checking if (s +d) is even, and (i,j) = ((s +d)/2, (s -d)/2) is valid, and i not in R, j not in C.

Thus, the code for C is:

C = 0

for s in S:

    for d in D:

        if (s + d) % 2 != 0:

            continue

        i = (s + d) // 2

        j = (s - d) // 2

        if 1 <= i <= N and 1 <= j <= N:

            if i not in R and j not in C:

                C +=1

Thus, the code can be implemented.

Thus, the final code is:

Read N and M.

Read M pairs, collect R, C, S, D as sets.

Compute initial = (N - len(R)) * (N - len(C)).

Compute A:

A = 0

for s in S:

    a = max(1, s - N)

    b = min(N, s - 1)

    if a > b:

        continue

    # Compute num_i_in_R

    num_i_in_R = sum(1 for i in R if a <= i <= b)

    # Compute num_j_in_C: j ranges a to b

    num_j_in_C = sum(1 for j in C if a <= j <= b)

    # Compute num_i_in_R_and_j_in_C

    num_i_in_R_and_j_in_C = 0

    for i in R:

        if a <= i <= b:

            j = s - i

            if j in C:

                num_i_in_R_and_j_in_C +=1

    count = (b - a + 1) - (num_i_in_R + num_j_in_C - num_i_in_R_and_j_in_C)

    A += count

Compute B:

B = 0

for d in D:

    a = max(1, d + 1)

    b = min(N, N + d)

    if a > b:

        continue

    # Compute num_i_in_R

    num_i_in_R = sum(1 for i in R if a <= i <= b)

    # Compute j range: j = i -d, i ranges a to b → j ranges (a -d) to (b -d)

    j_min = a - d

    j_max = b - d

    # Clamp j range to 1..N

    j_min = max(1, j_min)

    j_max = min(N, j_max)

    if j_min > j_max:

        num_j_in_C = 0

    else:

        num_j_in_C = sum(1 for j in C if j_min <= j <= j_max)

    # Compute num_i_in_R_and_j_in_C

    num_i_in_R_and_j_in_C = 0

    for i in R:

        if a <= i <= b:

            j = i - d

            if j >=1 and j <=N and j in C:

                num_i_in_R_and_j_in_C +=1

    count = (b - a + 1) - (num_i_in_R + num_j_in_C - num_i_in_R_and_j_in_C)

    B += count

Compute C:

C = 0

for s in S:

    for d in D:

        if (s + d) % 2 != 0:

            continue

        i = (s + d) // 2

        j = (s - d) // 2

        if 1 <= i <= N and 1 <= j <= N:

            if i not in R and j not in C:

                C +=1

valid = initial - (A + B - C)

print(valid)

Now, let's test this code with sample input 1.

Sample input 1:

N=8, M=6

R = {1,2,3,4,5,8}

C = {1,2,3,4,5,8}

S = {3,5,7,9,11}

D = {-5, -3, -1,1,3,5}

initial = (8-6)*(8-6)=2*2=4.

A:

For s in S:

s=3:

a=1, b=2.

num_i_in_R = sum(1 for i in R if 1<=i <=2 → i=1,2 → 2.

num_j_in_C = sum(1 for j in C if 1<=j <=2 → j=1,2 →2.

num_i_in_R_and_j_in_C:

for i in R → i=1,2,3,4,5,8.

i=1: j=3-1=2 → in C → yes. → count +=1.

i=2: j=3-2=1 → in C → yes. → count +=1.

others not in a<=i <=b.

Thus, num_i_in_R_and_j_in_C=2.

count = (2-1+1) - (2+2-2) →2 -2=0.

A +=0.

s=5:

a=1, b=4.

num_i_in_R = sum(1 for i in R if 1<=i <=4 → i=1,2,3,4 →4.

num_j_in_C = sum(1 for j in C if 1<=j <=4 → j=1,2,3,4 →4.

num_i_in_R_and_j_in_C:

i=1: j=4 → in C → yes.

i=2: j=3 → in C → yes.

i=3: j=2 → in C → yes.

i=4: j=1 → in C → yes.

→4.

count = (4-1+1) - (4+4-4) →4 -4=0.

A +=0.

s=7:

a=1, b=6.

num_i_in_R = sum(1 for i in R if 1<=i <=6 → i=1,2,3,4,5 →5.

num_j_in_C = sum(1 for j in C if 1<=j <=6 → j=1,2,3,4,5 →5.

num_i_in_R_and_j_in_C:

i=1: j=6 → not in C.

i=2: j=5 → in C → yes.

i=3: j=4 → in C → yes.

i=4: j=3 → in C → yes.

i=5: j=2 → in C → yes.

→4.

count =6 - (5+5-4) =6-6=0.

A +=0.

s=9:

a=1, b=8.

num_i_in_R = sum(1 for i in R if 1<=i <=8 →6.

num_j_in_C = sum(1 for j in C if 1<=j <=8 →6.

num_i_in_R_and_j_in_C:

for i in R:

i=1: j=8 → in C → yes.

i=2: j=7 → not in C.

i=3: j=6 → not in C.

i=4: j=5 → in C → yes.

i=5: j=4 → in C → yes.

i=8: j=1 → in C → yes.

→4.

count =8 - (6+6-4) =8-8=0.

A +=0.

s=11:

a=3, b=8.

num_i_in_R = sum(1 for i in R if 3<=i <=8 →3,4,5,8 →4.

num_j_in_C = sum(1 for j in C if 3<=j <=8 →3,4,5,8 →4.

num_i_in_R_and_j_in_C:

i=3: j=8 → in C → yes.

i=4: j=7 → not in C.

i=5: j=6 → not in C.

i=8: j=3 → in C → yes.

→2.

count =6 - (4+4-2) =6-6=0.

A +=0.

Thus, A=0.

B:

For d in D:

d=-5:

a= max(1, -5+1)=1.

b= min(8, 8+(-5))=3.

j =i - (-5) =i+5.

j ranges from 1+5=6 to 3+5=8.

num_i_in_R = sum(1 for i in R if 1<=i <=3 →i=1,2,3 →3.

num_j_in_C = sum(1 for j in C if 6<=j <=8 →j=8 →1.

num_i_in_R_and_j_in_C:

i=1 →j=6 → not in C.

i=2 →j=7 → not in C.

i=3 →j=8 → in C → yes.

→1.

count =3 - (3+1-1) =3-3=0.

B +=0.

d=-3:

a= max(1, -3+1)=1.

b= min(8, 8+(-3))=5.

j ranges i+3.

i ranges 1-5.

j ranges 4-8.

But j ranges 4-8.

num_i_in_R = sum(1 for i in R if 1<=i <=5 →i=1,2,3,4,5 →5.

num_j_in_C = sum(1 for j in C if 4<=j <=8 →j=4,5,8 →3.

num_i_in_R_and_j_in_C:

i=1 →j=4 → in C → yes.

i=2 →j=5 → in C → yes.

i=3 →j=6 → not in C.

i=4 →j=7 → not in C.

i=5 →j=8 → in C → yes.

→3.

count =5 - (5+3-3) =5-5=0.

B +=0.

d=-1:

a= max(1, -1+1)=1.

b= min(8,8 +(-1))=7.

j =i +1.

i ranges 1-7.

j ranges 2-8.

num_i_in_R = sum(1 for i in R if 1<=i <=7 →i=1,2,3,4,5 →5.

num_j_in_C = sum(1 for j in C if 2<=j <=8 →j=2,3,4,5,8 →5.

num_i_in_R_and_j_in_C:

i=1 →j=2 → in C → yes.

i=2 →j=3 → in C → yes.

i=3 →j=4 → in C → yes.

i=4 →j=5 → in C → yes.

i=5 →j=6 → not in C.

→4.

count =7 - (5+5-4) =7-6=1.

B +=1.

d=1:

a= max(1,1+1)=2.

b= min(8,8+1)=8.

j =i -1.

i ranges 2-8.

j ranges 1-7.

num_i_in_R = sum(1 for i in R if 2<=i <=8 →i=2,3,4,5,8 →5.

num_j_in_C = sum(1 for j in C if 1<=j <=7 →j=1,2,3,4,5 →5.

num_i_in_R_and_j_in_C:

i=2 →j=1 → in C → yes.

i=3 →j=2 → in C → yes.

i=4 →j=3 → in C → yes.

i=5 →j=4 → in C → yes.

i=8 →j=7 → not in C.

→4.

count =7 - (5+5-4) =7-6=1.

B +=1.

d=3:

a= max(1,3+1)=4.

b= min(8,8+3)=8.

j =i -3.

i ranges 4-8.

j ranges 1-5.

num_i_in_R = sum(1 for i in R if 4<=i <=8 →4,5,8 →3.

num_j_in_C = sum(1 for j in C if 1<=j <=5 →1,2,3,4,5 →5.

num_i_in_R_and_j_in_C:

i=4 →j=1 → in C → yes.

i=5 →j=2 → in C → yes.

i=8 →j=5 → in C → yes.

→3.

count =5 - (3+5-3) =5-5=0.

B +=0.

d=5:

a= max(1,5+1)=6.

b= min(8,8+5)=8.

j =i -5.

i ranges 6-8.

j ranges 1-3.

num_i_in_R = sum(1 for i in R if 6<=i <=8 →8 →1.

num_j_in_C = sum(1 for j in C if 1<=j <=3 →1,2,3 →3.

num_i_in_R_and_j_in_C:

i=8 →j=3 → in C → yes.

→1.

count =3 - (1+3-1) =3-3=0.

B +=0.

Thus, B=1+1=2.

C:

C=0.

For each s in S and d in D:

s=3, d=-5:

s +d =-2 → even.

i= (-2)/2 =-1 → invalid.

s=3, d=-3 → sum 0 → even.

i=0 → invalid.

s=3, d=-1 → sum 2 → even.

i=1, j=2.

i=1 is in R → no.

s=3, d=1 → sum 4 → even.

i=2, j=1.

i=2 is in R → no.

s=3, d=3 → sum 6 → even.

i=3, j=0 → invalid.

s=3, d=5 → sum 8 → even.

i=4, j=-1 → invalid.

s=5, d=-5 → sum 0 → even.

i=0 → invalid.

s=5, d=-3 → sum 2 → even.

i=1, j=4.

i=1 is in R → no.

s=5, d=-1 → sum 4 → even.

i=2, j=3.

i=2 is in R → no.

s=5, d=1 → sum 6 → even.

i=3, j=2.

i=3 is in R → no.

s=5, d=3 → sum 8 → even.

i=4, j=1.

i=4 is in R → no.

s=5, d=5 → sum 10 → even.

i=5, j=0 → invalid.

s=7, d=-5 → sum 2 → even.

i=1, j=8.

i=1 is in R → no.

s=7, d=-3 → sum 4 → even.

i=2, j=5.

i=2 is in R → no.

s=7, d=-1 → sum 6 → even.

i=3, j=4.

i=3 is in R → no.

s=7, d=1 → sum 8 → even.

i=4, j=3.

i=4 is in R → no.

s=7, d=3 → sum 10 → even.

i=5, j=2.

i=5 is in R → no.

s=7, d=5 → sum 12 → even.

i=6, j=1.

i not in R, j=1 is in C → no.

s=9, d=-5 → sum 4 → even.

i=2, j=7.

i=2 is in R → no.

s=9, d=-3 → sum 6 → even.

i=3, j=6.

i=3 is in R → no.

s=9, d=-1 → sum 8 → even.

i=4, j=5.

i=4 is in R → no.

s=9, d=1 → sum 10 → even.

i=5, j=4.

i=5 is in R → no.

s=9, d=3 → sum 12 → even.

i=6, j=3.

i not in R, j=3 is in C → no.

s=9, d=5 → sum 14 → even.

i=7, j=2.

i not in R, j=2 is in C → no.

s=11, d=-5 → sum 6 → even.

i=3, j=8.

i=3 is in R → no.

s=11, d=-3 → sum 8 → even.

i=4, j=7.

i=4 is in R → no.

s=11, d=-1 → sum 10 → even.

i=5, j=6.

i=5 is in R → no.

s=11, d=1 → sum 12 → even.

i