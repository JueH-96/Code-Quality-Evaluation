def main():
    import sys

    data = sys.stdin.read().strip().split()
    if not data: 
        return
    t = int(data[0])
    answers = []
    # Here is how we can think about the problem:
    #
    # There are N equally spaced points on a circle. In each move, the current operator (Alice at 0 or Bob at K)
    # must color exactly one white point black, but after the move the overall black/white coloring of the N points
    # must be symmetric “about” the line joining the operator’s position and the center.
    #
    # For a given operator at position P, let the reflection of any point x be given by:
    #       r_P(x) = 2*P - x  (mod N)
    # In order for the overall coloring to be symmetric with respect to this reflection, every pair of symmetric points 
    # must have the same color.
    #
    # When we color one white point black (say x), there are two possibilities:
    #    1) x is fixed under r_P, i.e. r_P(x) = x. In this case the symmetry is not disturbed.
    #    2) x is not fixed, so r_P(x) != x. But then before the move, both x and r_P(x) must be white/implying that if
    #       we only change one to black, the symmetry would be lost. Hence the rule forces that in such a move the chosen x
    #       must have r_P(x) already black, so that post move both are black.
    # However, note that initially all points are white. Thus on the very first move, the only allowed move is to color a fixed point.
    #
    # For a given reflection r_P the fixed points x satisfy:
    #      x = 2*P - x  (mod N)  
    #  =>   2x = 2*P  (mod N)  
    #  =>   x = P  if N is odd, but if N is even, then the congruence 2x ≡ 2P (mod N) has exactly two solutions:
    #         x = P and x = P + N/2 (mod N).
    #
    # Thus:
    #  - When N is odd:
    #         Alice's reflection (from point 0) has a unique fixed point: 0.
    #         Bob's reflection (from point K) has a unique fixed point: K.
    #         So on the first move, Alice can only color point 0 and on Bob’s turn he can only color point K.
    #         Then the game stalls (although later pairing moves might be possible if some partner was already black, 
    #         here with all white initially the game would end with only 2 points colored).
    #         Hence, it is impossible to color all points when N is odd.
    #
    #  - When N is even:
    #         Alice (at 0) has two fixed points: 0 and N/2.
    #         Bob (at K) has two fixed points: K and (K+N/2) mod N.
    #         Notice that if Bob’s fixed points coincide with Alice’s (i.e. {K, K+N/2} equals {0, N/2}),
    #         then the players face an overlap meaning that they “compete” for the same fixed points in the early moves.
    #         In fact, one can show (and the samples confirm) that if K equals N/2 then not all points can be colored.
    #         Otherwise, the players can plan their moves (using fixed‐point moves initially and later “pairing” moves)
    #         in a cooperative strategy to eventually color all points.
    #
    # Therefore, the answer is "Yes" if and only if N is even and K != N/2, and "No" otherwise.

    j = 1
    for _ in range(t):
        N = int(data[j]); K = int(data[j+1]); j += 2
        if N % 2 == 1:
            answers.append("No")
        else:
            if K == N // 2:
                answers.append("No")
            else:
                answers.append("Yes")
                
    sys.stdout.write("
".join(answers))

if __name__ == '__main__':
    main()