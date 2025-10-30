import sys
import collections

# It is good practice to read large inputs using sys.stdin.readline
# for performance reasons in a competitive programming context.
N, M = map(int, sys.stdin.readline().split())
S = sys.stdin.readline().rstrip()
T = sys.stdin.readline().rstrip()

# mismatch_count[j] stores the number of differing characters
# between the substring S[j:j+M] and the pattern T.
mismatch_count = [0] * (N - M + 1)

# j_depends_on_i[i] stores a list of segment starting indices 'j'
# that have a mismatch at position 'i'. This represents a reverse
# dependency: for segment 'j' to become fully matched, the mismatch
# at 'i' must be resolved by another operation.
j_depends_on_i = [[] for _ in range(N)]

# Pre-computation of mismatches and dependencies.
# This loop iterates through all possible start positions 'j' for T.
for j in range(N - M + 1):
    # For each start position 'j', compare S[j:j+M] with T.
    for p in range(M):
        i = j + p
        if S[i] != T[p]:
            mismatch_count[j] += 1
            j_depends_on_i[i].append(j)

# A queue for segments that are ready to be "explained" or "resolved".
# A segment is ready if its mismatch_count is 0.
q = collections.deque()

# A boolean array to prevent adding the same segment index to the queue multiple times.
in_queue = [False] * (N - M + 1)

# Initialize the queue with segments that are perfect matches from the start.
for j in range(N - M + 1):
    if mismatch_count[j] == 0:
        if not in_queue[j]:
            q.append(j)
            in_queue[j] = True

# Keep track of how many characters in S have been successfully "matched".
matched_count = 0
# A boolean array to track which character positions are matched.
matched = [False] * N

# Process the queue until it's empty (a worklist algorithm).
while q:
    k = q.popleft()
    
    # For the resolved segment k, mark its constituent characters as matched.
    for p in range(M):
        i = k + p
        if not matched[i]:
            matched[i] = True
            matched_count += 1
            
            # For each segment 'j' that had a mismatch at 'i', decrement its
            # mismatch count as this position is now considered resolved.
            for j in j_depends_on_i[i]:
                mismatch_count[j] -= 1
                # If a segment's mismatch count drops to 0, it's now fully
                # explainable and ready to be processed. Add it to the queue.
                if mismatch_count[j] == 0 and not in_queue[j]:
                    q.append(j)
                    in_queue[j] = True

# If all N characters were matched, the string S is constructible.
if matched_count == N:
    print("Yes")
else:
    print("No")