def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N, S = map(int, input_data[:2])
    A = list(map(int, input_data[2:]))

    # Quick checks
    # 1) If any A_i == S, answer Yes immediately
    # 2) If min(A) > S, answer No immediately
    # 3) If sum(A) == S, answer Yes (because one full period sums to S)
    # These are not strictly necessary for correctness (the general
    # two-pointer-with-skipping will also catch them), but they can
    # short-circuit some easy cases.
    P = sum(A)
    if S in A:  # Single element equals S
        print("Yes")
        return
    if min(A) > S:  # All elements too large
        print("No")
        return
    if P == S:  # One full period matches S
        print("Yes")
        return

    # We will use a two-pointer–style approach on the infinite, repeating array.
    # However, we cannot literally build an infinite array. Instead, we:
    #
    # 1) Keep track of the current subarray sum (sumSoFar).
    # 2) Have left and right indices (l, r) that conceptually traverse
    #    the infinite sequence by using modulo to pick elements from A.
    # 3) Whenever sumSoFar < S but adding an entire period sum P still stays ≤ S,
    #    we skip as many full periods as possible in one go (to move r forward in O(1) time).
    # 4) Otherwise we advance r by one element (r += 1) or move l by one (l += 1) to shrink.
    # 5) We stop if the subarray (r - l) becomes too large (larger than some multiple of N)
    #    without finding S, because by that point further repetition won’t produce anything new
    #    that we haven’t effectively already tried.
    #
    # A known safe bound (for positive arrays) is that we only need to allow subarray lengths
    # up to about 3*N to capture crossing up to two period boundaries. If we exceed that
    # window without finding sum == S, we can conclude "No".
    #
    # We also safeguard the total number of expansions (increments of r or l) to ensure
    # we do not loop too long. In the worst case, we allow up to ~6*N expansions.

    def get_elem(idx):
        # Element at 'idx' in the infinite repetition
        return A[idx % N]

    max_len = 3 * N  # maximum length of the window r - l
    sumSoFar = 0
    l = 0
    r = 0

    # Count how many pointer moves we've done, to avoid infinite looping
    moves = 0
    moves_limit = 6 * N  # enough slack for expansions/contractions

    while moves < moves_limit:
        if sumSoFar < S:
            # See if we can skip whole periods quickly
            if sumSoFar + P <= S:
                # We can skip at least one entire period
                # but do not exceed the max_len window
                can_skip = (S - sumSoFar) // P  # how many full periods we *could* skip
                allowed = max_len - (r - l)
                max_whole = allowed // N  # how many full periods we *may* skip without exceeding window

                skip_count = min(can_skip, max_whole)
                if skip_count > 0:
                    sumSoFar += skip_count * P
                    r += skip_count * N
                    moves += 1  # count this as one "move" for bulk skipping
                    if sumSoFar == S:
                        print("Yes")
                        return
                    continue
                else:
                    # Can't skip a whole period (would overshoot or exceed window),
                    # so advance by one element
                    if (r - l) >= max_len:
                        # Window is already at max, no point in adding one more
                        break
                    sumSoFar += get_elem(r)
                    r += 1
                    moves += 1
                    if sumSoFar == S:
                        print("Yes")
                        return
            else:
                # sumSoFar + P > S, so skip zero periods, just advance r by 1
                if (r - l) >= max_len:
                    break
                sumSoFar += get_elem(r)
                r += 1
                moves += 1
                if sumSoFar == S:
                    print("Yes")
                    return

        elif sumSoFar > S:
            # Shrink from the left
            if l < r:
                sumSoFar -= get_elem(l)
                l += 1
                moves += 1
                if sumSoFar == S:
                    print("Yes")
                    return
            else:
                # If somehow l == r but sumSoFar > S with positive A,
                # then we've got no valid window. Reset.
                sumSoFar = 0

        else:
            # sumSoFar == S
            print("Yes")
            return

        # Check window size
        if (r - l) > max_len:
            break

    # If we exit the loop, we failed to find a subarray summing to S
    print("No")

# Call main() at the end
if __name__ == "__main__":
    main()