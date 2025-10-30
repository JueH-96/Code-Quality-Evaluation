import sys

def solve():
    """
    Solves the problem by reading input, using prefix sums for pre-computation,
    and then answering each query in O(1) time.
    """
    # Use fast I/O for large inputs
    input = sys.stdin.readline

    # Read problem size and number of queries
    try:
        N, Q = map(int, input().split())
    except ValueError:
        # Handle case where input line might be empty
        return

    # Read the string
    S = input().strip()

    # --- Pre-computation Step ---
    # Create a prefix sum array `prefix_counts`.
    # `prefix_counts[i]` will store the number of adjacent identical characters
    # in the prefix S[0...i]. That is, the number of indices j < i where S[j] == S[j+1].
    prefix_counts = [0] * N

    # `prefix_counts[0]` is 0 as a single character prefix has no pairs.
    for i in range(1, N):
        # Start with the count from the previous prefix
        prefix_counts[i] = prefix_counts[i-1]
        # If the new character S[i] is same as the previous one S[i-1],
        # we have one more adjacent pair.
        if S[i-1] == S[i]:
            prefix_counts[i] += 1

    # --- Query Processing Step ---
    answers = []
    for _ in range(Q):
        l, r = map(int, input().split())

        # The query is for the 1-based range [l, r].
        # We need to count integers p such that l <= p <= r-1 and S_p = S_{p+1}.
        # In 0-based indexing, this corresponds to indices j = p-1.
        # So, we count j in the range [l-1, r-2] where S[j] == S[j+1].
        
        # The number of pairs in the range [l-1, r-2] is:
        # (total pairs up to index r-2) - (total pairs up to index l-2).
        
        # From our `prefix_counts` definition:
        # Total pairs up to index k-1 (i.e., j < k) is `prefix_counts[k]`.
        # So, total pairs up to r-2 is `prefix_counts[r-1]`.
        # And total pairs up to l-2 is `prefix_counts[l-1]`.
        
        # The result is the difference.
        # Note: 1-based indices l, r are converted to 0-based by subtracting 1.
        result = prefix_counts[r - 1] - prefix_counts[l - 1]
        answers.append(str(result))

    # Print all answers at once for efficiency.
    print("
".join(answers))

solve()