def main():
    import sys
    input_data = sys.stdin.buffer.read().split()
    if not input_data:
        return

    it = iter(input_data)
    N = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    
    # Explanation:
    # Each alien starts with A[i] stones at time of turning adult.
    # When an alien turns adult, it instantly receives a “gift” from every adult who was already available and had at least one stone.
    # After turning adult, in each future year such an adult (donor) will give one stone to the new adult if it still has a stone.
    # Note that the number of times an adult gives stones is determined by its stone count when it became adult,
    # but that stone count for donors can also be increased by a gift on the very year they turn adult.
    #
    # We can “simulate” the process with the following observation:
    #  1. Let curr[i] be the number of stones alien i has immediately after turning adult.
    #     For alien i (0-indexed) we have: curr[i] = A[i] + (gifts received when i turned adult).
    #     Note that alien 0 (i=0) turns adult first so gets no gifts.
    #
    #  2. In each later year, if an alien has at least one stone then it “donates” one stone.
    #     Thus, alien i (turning adult at year i+1) will donate in each future donation event until its stones run out.
    #     The maximum times it can donate is the number of future events (N - 1 - i).
    #
    #  3. How many times does it actually donate? Exactly:
    #         donate[i] = min(curr[i], N - 1 - i)
    #
    #  4. Its final stone count then becomes:
    #         final[i] = curr[i] - donate[i]
    #
    #  5. Moreover, each donation by an alien i gives a gift of +1 stone to a future alien.
    #     More precisely, if donor i makes donate[i] donations, then every alien turning adult in the years
    #     from (i+1) to (i + donate[i]) gets one extra stone.
    #
    # We can accumulate these contributions with a difference‐array.
    #
    # Implementation details:
    #  - Use a diff array of length N+1 (0-indexed) where diff[j] will be added to get the total gifts
    #    that the alien turning adult at year j+1 receives.
    #  - Process aliens in order. For i from 0 to N-1:
    #       • Compute gift_received = cumulative sum from diff at index i.
    #       • Set curr = A[i] + gift_received.
    #       • Donation events available in the future = N - 1 - i.
    #       • Compute donate = min(curr, future_events).
    #       • The alien’s final stone count becomes curr - donate.
    #       • Then update the diff array to add +1 for all aliens in range [i+1, i+donate].
    
    diff = [0] * (N + 1)
    result = [0] * N
    cum = 0  # cumulative gifts from diff
    for i in range(N):
        cum += diff[i]
        curr = A[i] + cum
        # Future donation events available (for aliens turning adult after i)
        future = N - 1 - i
        donate = curr if curr < future else future
        result[i] = curr - donate
        if donate > 0:
            start = i + 1
            end = i + donate + 1  # end index is exclusive
            diff[start] += 1
            if end < len(diff):
                diff[end] -= 1

    sys.stdout.write(" ".join(map(str, result)))

if __name__ == '__main__':
    main()