import sys

def solve():
    N = int(sys.stdin.readline())
    P = list(map(int, sys.stdin.readline().split()))

    if N == 1:
        print(0)
        return

    p1_score = P[0]
    
    # Scores of other people (persons 2 to N).
    # P[1:] is a slice containing these scores.
    # Since N > 1, this slice is guaranteed to be non-empty.
    other_scores = P[1:]
    max_other_score = max(other_scores)
    
    # Person 1's new score (p1_score + x) must be strictly greater than max_other_score.
    # Let S_new = p1_score + x.
    # We need S_new > max_other_score.
    # Since scores are integers, this is equivalent to S_new >= max_other_score + 1.
    # The minimum integer value for S_new is max_other_score + 1.
    
    # To find the minimum x, we set p1_score + x equal to this target minimum new score:
    # p1_score + x = max_other_score + 1
    # x = (max_other_score + 1) - p1_score
    
    points_to_add = (max_other_score + 1) - p1_score
    
    # The problem states x must be non-negative.
    # If points_to_add is negative, it means p1_score was already sufficiently high
    # (i.e., p1_score >= max_other_score + 1, or p1_score > max_other_score).
    # In this case, no points need to be added, so x = 0.
    # The max(0, points_to_add) handles this correctly.
    
    result = max(0, points_to_add)
    
    print(result)

if __name__ == '__main__':
    solve()