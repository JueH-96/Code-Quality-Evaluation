import sys

def solve():
    # Read input N
    N = int(sys.stdin.readline())
    # Read heights H_1, H_2, ..., H_N
    # Use 0-based indexing internally for H: H = [H_0, H_1, ..., H_{N-1}]
    # The problem refers to buildings 1 to N, corresponding to indices 0 to N-1.
    H = list(map(int, sys.stdin.readline().split()))

    # Compute L_j for j = 0, ..., N-1 (0-based index)
    # L_j is the largest index k < j such that H[k] > H[j].
    # If no such k exists, L_j is -1.
    # We use a monotonic decreasing stack of indices.
    # The stack will store indices k such that H[k] is strictly decreasing.
    L = [-1] * N
    stack = [] # Stores 0-based indices k in increasing order such that H[k] is decreasing

    for j in range(N):
        # While the stack is not empty and the height of the building at the top of the stack
        # is less than the current building's height H[j], pop from the stack.
        # These popped buildings cannot be the first taller building to the left of any future index j' > j
        # because H[j] is now in between and is taller.
        while stack and H[stack[-1]] < H[j]:
            stack.pop()

        # After popping, if the stack is not empty, the index at the top is the first index k < j
        # such that H[k] > H[j]. This is our L_j.
        if stack:
            L[j] = stack[-1]
        else:
            # If the stack is empty, it means no building to the left of j is taller than H[j].
            # We represent this with L_j = -1 (using 0-based indexing, where -1 is before the first index 0).
            L[j] = -1

        # Push the current index j onto the stack.
        stack.append(j)

    # Now we have L_j for all j = 0, ..., N-1.
    # For each building i (0-based), we need to count the number of buildings j (0-based)
    # such that i < j <= N-1 and there is no building k with i < k < j (0-based) taller than H[j].
    # The condition "no building k with i < k < j taller than H[j]" is equivalent to:
    # the index of the largest k < j with H[k] > H[j] must be less than or equal to i.
    # In our 0-based indexing, this index is L_j.
    # So, for a fixed j, it contributes to c_i if and only if i < j and L_j <= i.
    # The valid range for i (0-based) is [max(0, L_j), j-1].

    # We need to calculate c_i for each i from 0 to N-1.
    # For each j from 0 to N-1, we increment the count for c_i for all i in the range [max(0, L_j), j-1].
    # We can do this efficiently using a difference array.
    # diff[k] stores the amount to add starting from index k.
    # The size of diff array should be N+1 to handle the end boundary condition for index N-1.
    # To increment values in the range [s, e], we do diff[s]++ and diff[e+1]--.
    # Our range for i is [max(0, L_j), j-1].

    diff = [0] * (N + 1) # Using 0-based index for diff, size N+1

    for j in range(N):
        # Range of i (0-based) that j contributes to: [max(0, L_j), j-1]
        start_i = max(0, L[j])
        end_i = j - 1

        # If the range is valid (start_i <= end_i)
        if start_i <= end_i:
            diff[start_i] += 1
            # The range ends at end_i, so we subtract 1 at index end_i + 1
            diff[end_i + 1] -= 1

    # Compute the final counts c_i by calculating prefix sums of the diff array.
    # c[i] will be the sum diff[0] + diff[1] + ... + diff[i].
    # The difference array property is that the prefix sum up to index k gives the value at index k
    # when the difference array represents point values.
    # Here, diff represents increments over ranges. The value at index i is the total increment
    # up to index i.

    result_c = [0] * N # Stores c_0, c_1, ..., c_{N-1}

    current_increment_sum = 0
    for i in range(N):
        current_increment_sum += diff[i]
        result_c[i] = current_increment_sum

    # Print the results separated by spaces.
    # The result_c list stores c_0, c_1, ..., c_{N-1}, which correspond to
    # the problem's required output c_1, c_2, ..., c_N for buildings 1 to N.
    print(*result_c)

solve()