def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    hs = list(map(int, data[1:]))
    
    # Precomputed "extra" damage for the first one or two attacks given a starting T mod 3.
    # Let r = current T mod 3. When attacking an enemy, the damage pattern (when the enemy is attacked consecutively)
    # is determined by the sequence of turns.
    # For any full block of 3 consecutive attacks, the damage sum is always 1+1+3 = 5.
    #
    # But the first few attacks (if not a multiple of 3) depend on the starting offset.
    #
    # If current T mod 3 = 0, then the upcoming attack turns are:
    #   Attack1: turn T+1, (0+1 mod 3 = 1) -> damage 1
    #   Attack2: turn T+2, (0+2 mod 3 = 2) -> damage 1
    #   Attack3: turn T+3, (0+3 mod 3 = 0) -> damage 3
    # So extra damage for 0,1,2 extra attacks are: [0, 1, 2]
    #
    # Similarly if current T mod 3 = 1:
    #   Attack1: (1+1=2 mod 3) -> damage 1
    #   Attack2: (1+2=3 mod 3) -> damage 3
    #   Attack3: (1+3=4 mod 3 = 1) -> damage 1
    # Extra damage: [0, 1, 4]
    #
    # And if current T mod 3 = 2:
    #   Attack1: (2+1=3 mod 3) -> damage 3
    #   Attack2: (2+2=4 mod 3 = 1) -> damage 1
    #   Attack3: (2+3=5 mod 3 = 2) -> damage 1
    # Extra damage: [0, 3, 4]
    extras = [
        [0, 1, 2],  # for r == 0
        [0, 1, 4],  # for r == 1
        [0, 3, 4]   # for r == 2
    ]
    
    # Global time counter T 
    total_time = 0
    # Process each enemy in order. When we start attacking an enemy,
    # our current global time (and therefore T mod 3) is fixed.
    # Then, enemy i with health h is attacked consecutively until its health is 0 or less.
    # In each full block of 3 consecutive attacks, the damage is 5.
    # For a given enemy, if we let n be the number of attacks required and 
    # denote n = 3*q + rem, where rem = 0, 1, or 2, then the total 
    # damage = 5*q + extra_damage, where extra_damage depends on rem and on the starting T mod 3.
    # We then choose the minimal n (i.e. minimal candidate = 3*q + rem) satisfying:
    #   5*q + extra[r][rem] >= h.
    # After finishing one enemy, T is increased by the number of attacks spent.
    for h in hs:
        r = total_time % 3  # starting remainder for the enemy
        best = None
        # Try each possibility for the leftover (rem = 0, 1, or 2) attacks
        for rem in range(3):
            extra_damage = extras[r][rem]
            # We need 5*q + extra_damage >= h.
            # If h is already less than or equal to extra_damage, we can choose q = 0.
            if h <= extra_damage:
                q = 0
            else:
                # Otherwise, compute the minimal q with ceiling division.
                q = (h - extra_damage + 4) // 5
            candidate = 3 * q + rem
            if best is None or candidate < best:
                best = candidate
        total_time += best

    sys.stdout.write(str(total_time))


if __name__ == '__main__':
    main()