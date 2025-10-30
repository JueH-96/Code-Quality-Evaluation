def solve_enemy(H, start_mod):
    # Given an enemy with health H and that the first attack on it happens
    # when the global turn number A satisfies A % 3 == start_mod,
    # we need to compute the minimum number of attacks t (t >= 1) required
    # so that the total damage delivered is at least H.
    #
    # On each turn we do one attack; the damage depends on the turn number:
    #   If T (the global turn) is a multiple of 3, damage = 3; otherwise damage = 1.
    #
    # Since the attacks on one enemy are consecutive, the pattern of T-values is:
    #   A, A+1, A+2, ..., A+t-1.
    #
    # Notice that in any block of 3 consecutive turns, no matter what A mod 3 is,
    # the total damage is always 5. (Because:
    #    - if A % 3 == 0, then turns: 3, 1, 1 give 5;
    #    - if A % 3 == 1, then turns: 1, 1, 3 give 5;
    #    - if A % 3 == 2, then turns: 1, 3, 1 give 5.)
    #
    # Let t = 3*q + r, where 0 <= r < 3.
    # Then the damage equals:
    #    Damage = 5*q + (extra damage in the r partial turns)
    #
    # The extra damage in the r extra turns (which come from a block starting with start_mod)
    # is as follows:
    #   • For r = 0: extra = 0.
    #   • For r = 1: 
    #         if start_mod == 0 then the attack is on a turn that is a multiple of 3,
    #         so extra = 3;
    #         otherwise extra = 1.
    #   • For r = 2:
    #         if start_mod == 0 then extra = 4   (first turn: 3, second: 1).
    #         if start_mod == 1 then extra = 2   (first: 1, second: 1).
    #         if start_mod == 2 then extra = 4   (first: 1, second: 3).
    #
    # We want the smallest t so that:
    #    5*q + extra >= H.
    #
    # We try each possibility for r = 0, 1, 2 and compute the minimal q required.
    
    extras = [
        [0, 0, 0],  # for r = 0, extra = 0 for any start_mod
        [3, 1, 1],  # for r = 1: if start_mod==0 then 3; if 1 or 2 then 1.
        [4, 2, 4]   # for r = 2: for start_mod==0 ->4, for 1 ->2, for 2 ->4.
    ]
    best = None
    a = start_mod  # shorthand
    for r in range(3):
        extra = extras[r][a]
        # We need 5*q + extra >= H, i.e. 5*q >= (H - extra).
        # Let needed = H - extra.
        needed = H - extra
        if needed <= 0:
            q = 0
        else:
            # Ceiling division to find smallest nonnegative q with 5*q >= needed.
            q = (needed + 4) // 5
        t_candidate = 3 * q + r
        # Ensure we have at least one attack.
        if t_candidate < 1:
            t_candidate = 1
        if best is None or t_candidate < best:
            best = t_candidate
    return best

    
def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    n = int(data[0])
    # Read exactly n enemy healths.
    H_list = list(map(int, data[1:1+n]))
    
    total_turns = 0
    # Global turn T is maintained in total_turns.
    # For each enemy the first attack uses turn A = total_turns + 1.
    for H in H_list:
        A = total_turns + 1
        start_mod = A % 3
        # Compute the minimal number of attacks needed for this enemy.
        t_needed = solve_enemy(H, start_mod)
        total_turns += t_needed
    sys.stdout.write(str(total_turns))
    
if __name__ == '__main__':
    main()