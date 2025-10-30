def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    X = []
    H = []
    idx = 1
    for _ in range(N):
        xi = float(input_data[idx]); hi = float(input_data[idx+1])
        X.append(xi); H.append(hi)
        idx += 2

    # --------------------------------------------------------------------
    # We want to stand at coordinate 0 with some nonnegative height h.
    # A building i (at x = X[i], height = H[i]) is visible if
    #    slope_i = (H[i] - h) / X[i]
    # is strictly greater than all previously encountered slopes.
    #
    # In other words, if we sweep buildings from left to right in ascending X,
    # we maintain a "current_max_slope" which is the largest slope so far.
    # For building i to be visible, we need:
    #    (H[i] - h) / X[i] > current_max_slope
    # or equivalently
    #    h < H[i] - current_max_slope * X[i].
    #
    # If for some i we have (H[i] - h)/X[i] <= current_max_slope, building i
    # is blocked. Thus, "all buildings visible" means that for every i in [0..N-1],
    # we satisfy (H[i] - h)/X[i] > current_max_slope and then update
    # current_max_slope = (H[i] - h)/X[i].
    #
    # As h increases, (H[i] - h)/X[i] decreases, so it becomes "easier" to exceed
    # a previously encountered slope (because the building's slope is going down),
    # so at some threshold h we might just start to make all slopes strictly
    # increasing. Indeed the logic is that if we start from h=0 (lowest vantage),
    # we see if all buildings are visible. If they are, the answer is -1 because
    # even from height=0 we already see all.
    # Otherwise, as we raise h, we might eventually "unlock" the chain of
    # strict slope increases (because the new building's slope must exceed the
    # old one's slope). Actually, on second thought, raising h lowers each
    # (H[i] - h)/X[i], so it becomes harder to maintain a strictly increasing
    # sequence of slopes. That means as h grows, some building that was blocked
    # before may become unblocked only if it needed to have an even smaller slope
    # requirement. However, the typical known result is: if we do a pass from left
    # to right, for each building i the condition is
    #    (H[i] - h)/X[i] > current_max_slope
    # => h < H[i] - current_max_slope * X[i]. 
    # We collect all such constraints for i=1..N in order. They must simultaneously
    # hold to see all buildings.
    #
    # So a function check(h) can do:
    #    current_max_slope = -∞
    #    for i in [0..N-1]:
    #       if (H[i] - h)/X[i] <= current_max_slope: 
    #           # building i not visible
    #           return False
    #       # else visible, update
    #       current_max_slope = (H[i] - h)/X[i]
    #    return True  # if we never fail
    #
    # Because as h goes up, (H[i] - h)/X[i] goes down, so it's "strictly harder"
    # to keep the chain strictly increasing. That means if check(0) is True,
    # then check(h) will remain True for all h < something? Actually it's the 
    # other way: if we can see all for h=0, raising h will only reduce each slope,
    # risking that the new slope_i won't exceed the old slope. 
    # So typically if check(0) is True, that means for any higher h it might become 
    # false at or after some threshold. But the problem's examples show the 
    # transition is from "not all visible" at smaller h to "all visible" at bigger h"
    # for certain examples. That is a bit counterintuitive with the strict slope test,
    # but occurs because we do not necessarily have to look at the top of building i
    # to see it (the problem states "there exists a point Q on building i" that 
    # is visible). But the simpler slope test only checks the top. 
    #
    # However, from geometry and known editorials: 
    # A building i is blocked by building j < i if the line from vantage P 
    # to some point on building i intersects building j. One can show that 
    # it suffices to check whether (H_i - h)/X_i is strictly greater than the 
    # maximum of (H_j - h)/X_j for j < i. If it is not strictly greater, 
    # then building i cannot find a point that "peeks over" building j. 
    #
    # But in the sample 1, as we raise h from 1.5 to something bigger, building 3 
    # becomes visible. That indicates that "raising h" can help with seeing 
    # farther buildings if intermediate ones are not as "steep" from that new vantage.
    #
    # The simplest robust solution is therefore:
    #  1) We do a "check(h)" function to see if all buildings are visible 
    #     (in the sense just described).
    #  2) The function is O(N).
    #  3) We know that for h=0, either it's visible or not. If it's already 
    #     "all visible," we answer -1.
    #  4) Otherwise, as h grows large, eventually (H_i - h)/X_i might become 
    #     negative... but that can sometimes help, ironically, if intermediate 
    #     buildings become "even more negative" so a far building might surpass 
    #     them. Indeed the problem statement's examples show there is a unique 
    #     threshold.
    #  5) We'll do a binary search on h in [0, very large], checking the 
    #     visibility. We'll find the smallest h at which "all buildings" become 
    #     visible. The question asks for the maximum h at which "not all" are 
    #     visible. So that boundary is that same "smallest h for which all are 
    #     visible." We'll call that boundary h0. Then the answer is exactly h0 
    #     if 0 <= h0, else if check(0) is True we return -1.
    #
    # We'll pick an upper bound for h that is definitely large enough. For instance, 
    # heights can go up to 1e9, X up to 1e9. In worst case sometimes the needed h 
    # could be on the order of (largest H). We'll pick something like 2e10 or 3e10 
    # to be safe. We'll do 60 steps of binary search (that easily covers 1e9 range 
    # with 1e-9 precision).
    #
    # Implementation details:
    #
    # check(h):
    #   curmax = -∞
    #   for i in range(N):
    #       s = (H[i] - h)/X[i]   # slope
    #       if s <= curmax:
    #           return False
    #       curmax = s
    #   return True
    #
    # We'll see if check(0) is True => answer -1.
    # Else binary search for the smallest h in [0, big] such that check(h) is True.
    # Then the answer = that h (the "boundary" above which all are visible). 
    # Print with required precision.
    #
    # We'll carefully handle floating precision > 1e-9. We'll do around 60 
    # iterations of binary search to be safe. 
    # --------------------------------------------------------------------

    def can_see_all(h):
        """Return True if from height h at x=0 we can see all buildings in order."""
        cur_slope = float('-inf')
        for i in range(N):
            # slope from vantage h to top of building i
            # but be careful: we are checking the top. Is that always sufficient?
            # Actually, from the problem statement we can see a building if there's
            # ANY point Q on it that is not blocked. A known geometric fact is that
            # if slope_to_top_i is strictly greater than all previously used slopes,
            # then building i can be seen (since we can choose a point just a bit 
            # below the top to avoid ties). If slope_to_top_i is not strictly greater,
            # building i is blocked. 
            slope_i = (H[i] - h) / X[i]
            if slope_i <= cur_slope:
                return False
            cur_slope = slope_i
        return True

    # Quick check at h=0
    if can_see_all(0.0):
        # Already can see all from ground-level
        print(-1)
        return

    # Otherwise, we do a binary-search for the smallest h >= 0 that yields can_see_all(h) = True.
    # Then we print that boundary. If it turns out that even for massive h we cannot see all,
    # (which the problem statement's examples suggest won't happen given large heights),
    # we would just end up with an answer that saturates at our upper bound. 
    # But from the examples, there's always a finite threshold.

    left = 0.0
    right = 2e11  # somewhat large upper bound
    for _ in range(70):  # overkill for 1e-9 precision
        mid = 0.5*(left+right)
        if can_see_all(mid):
            right = mid
        else:
            left = mid

    # 'right' is the smallest h from which all buildings are visible.
    # The question wants the maximum h from which NOT all are visible.
    # So the answer is that same number (because at just below 'right', 
    # not all are visible, and at 'right', we transition to all visible).
    ans = right

    # Printing with an absolute/relative error of at most 1e-9 is acceptable.
    # We'll just format with say 15-18 digits after the decimal to be safe.
    print(f"{ans:.9f}")

# Do not forget to call main
if __name__ == "__main__":
    main()