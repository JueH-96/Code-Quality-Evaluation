def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))

    # We maintain a global "time" T, which starts at 0.
    # Each attack increments T by 1.
    # If T is a multiple of 3, the damage is 3; otherwise, the damage is 1.
    # We always attack the first enemy with health > 0, moving on once that enemy's health <= 0.
    # We want the final T when all enemies' health <= 0.

    # Direct simulation (attack-by-attack) is impossible for large inputs.
    # Instead, observe that to kill one enemy of health h, starting from a time-offset T0,
    # we need a certain number K of attacks. Over those K attacks, T goes from T0+1 to T0+K.
    #
    # The damage done in those K attacks is:
    #   - 3 points for each multiple of 3 in [T0+1, ..., T0+K],
    #   - 1 point otherwise.
    #
    # Let s = (T0+1) mod 3. Then the times in [T0+1..T0+K] have mod 3 values:
    #   s, (s+1)%3, (s+2)%3, ..., (s+K-1)%3.
    # The number of multiples of 3 in that set is the count of i in [0..K-1] such that (s + i) mod 3 = 0.
    # We define i0 = (3 - s) mod 3. Then those i that satisfy (s + i) mod 3 = 0 are i = i0, i0+3, i0+6, ...
    # as long as i <= K-1. So the count c(K) = max(0, floor((K-1 - i0)/3) + 1) if K>i0, else 0.
    #
    # Total damage = 3*c(K) + 1*(K - c(K)) = K + 2*c(K).
    #
    # We want the minimum K such that (K + 2*c(K)) >= h. Then we add that K to T, and proceed to the next enemy.
    #
    # Because c(K) can be computed in O(1), we can do a binary-search for K. The upper bound can be ~1.5*h or 2*h
    # to be safe, since purely 1-damage hits would require h attacks, and we can add some buffer.
    #
    # Implementation detail: We track T in a 64-bit sense (Python int is unbounded). Then for each enemy of health h_i:
    #   1) s = (T + 1) mod 3
    #   2) binary-search for minimal K where K+2*c(K) >= h_i
    #   3) T += K
    #   4) move to next enemy
    #
    # Finally print T.

    # Precompute function to get total damage with K attacks given offset s:
    def total_damage(K, i0):
        """Returns total damage from K consecutive attacks if the first attack index mod 3 = s,
           where i0 = (3 - s) % 3."""
        if K <= i0:
            # In these K attacks, none is a multiple of 3 (since the first multiple-of-3 index is i0)
            return K
        Kprime = K - i0  # after skipping i0 steps, from that point multiples-of-3 happen regularly
        # number of multiples of 3 among those K steps:
        # c = floor((Kprime-1)/3) + 1
        c = (Kprime - 1) // 3 + 1
        return K + 2 * c

    T = 0  # total time
    s = 1  # since T=0 => first attack is T=1 => T mod 3 = 1

    # For faster I/O, we will avoid too much overhead in loops.
    # We'll implement the binary search in a function.
    def needed_attacks(h, s):
        # i0 = the shift until we hit a multiple of 3 in the sequence
        i0 = (3 - s) % 3
        # We'll do a binary search for minimal K.
        # Upper bound can be up to 2*h (just to be safe).
        if h == 1:
            hi = 2  # small safe bound for h=1
        else:
            hi = 2 * h
        lo = 1
        # We can do a quick check: if lo + i0 >= h, that might suffice. But let's keep it simple.

        while lo < hi:
            mid = (lo + hi) // 2
            dmg = total_damage(mid, i0)
            if dmg >= h:
                hi = mid
            else:
                lo = mid + 1
        return lo

    for h_i in H:
        # Find how many attacks we need for this enemy's health h_i
        K = needed_attacks(h_i, s)
        T += K
        s = (s + K) % 3

    print(T)

# Call main() at the end.
main()