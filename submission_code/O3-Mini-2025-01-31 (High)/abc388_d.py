def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    # The first token is N, then N integers for A_i.
    N = int(data[0])
    A = list(map(int, data[1:1+N]))

    # We will process aliens in order (0-indexed corresponding to 1-indexed aliens).
    # When an alien matures, its stone count S is:
    #    S[i] = A[i] + (gifts it got from earlier adults)
    # Each mature alien will donate (in future rounds) exactly once in each subsequent year
    # provided it has a stone.
    # Because an alien (once adult) goes through exactly (N - 1 - i) donation rounds,
    # the maximum number of gifts it can give is (N - 1 - i) (the number of later events).
    # However, note that after maturing, an alien might receive gifts from earlier aliens.
    # Thus the actual donation count d[i] is:
    #    d[i] = min(S[i], N - 1 - i)
    # and its final stone count becomes:
    #    final[i] = S[i] - d[i]
    #
    # Moreover, when an alien i donates a stone in one of the subsequent rounds,
    # that donation is given to whichever alien matures in that corresponding round.
    # More precisely, alien i will donate in rounds:
    #    i+1, i+2, ..., i + d[i]
    # In each such round the receiving alien gets 1 stone.
    #
    # To efficiently simulate the distribution of gifts (donations),
    # we use a difference array "diff" which will allow us to add +1
    # to all aliens in a contiguous range.
    #
    # Let diff be an array of length N+1 (0-indexed).
    # We will maintain a running cumulative sum "gift_cum" so that when processing
    # alien i, the number of gifts it received from previous donation events is gift_cum.
    
    diff = [0] * (N + 1)
    result = [0] * N
    gift_cum = 0

    # Process aliens in order of maturing.
    for i in range(N):
        gift_cum += diff[i]
        # S[i]: stone count of alien i just after maturing (before it makes any donations)
        S = A[i] + gift_cum
        
        # Number of donation opportunities for alien i is the number of later events.
        opp = N - 1 - i
        d = S if S < opp else opp  # d = min(S, opp)
        
        # Final stones = current count minus stones donated.
        result[i] = S - d
        
        # Alien i will donate in rounds corresponding to maturing aliens:
        # from index i+1 up to and including index i+d.
        if d > 0:
            start = i + 1
            end = i + d + 1  # end is one-past the last index in range update.
            diff[start] += 1
            if end < len(diff):
                diff[end] -= 1

    # Output the final stones for each alien, space separated.
    sys.stdout.write(" ".join(map(str, result)))

if __name__ == '__main__':
    main()