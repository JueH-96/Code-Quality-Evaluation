def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))

    # Precompute the partial damage for each possible offset (the remainder of (T+1) mod 3)
    # offset=0 => next attack deals 3 damage, then 1, then 1
    # offset=1 => next attack deals 1 damage, then 1, then 3
    # offset=2 => next attack deals 1 damage, then 3, then 1
    # partial_damage[offset][r] = sum of the first r attacks in that sequence (r = 0..2)
    partial_damage = [
        [0, 3,  4],  # offset=0 => (3, 1, 1)
        [0, 1,  2],  # offset=1 => (1, 1, 3)
        [0, 1,  4]   # offset=2 => (1, 3, 1)
    ]

    T = 0  # The total number of attacks made so far
    for health in H:
        # The offset for the next attack is based on (T+1) % 3, because T is incremented before attacking.
        offset = (T + 1) % 3

        # We want the minimal L such that 5*(L//3) + partial_damage[offset][L%3] >= health
        # A quick way is to check M0 = health // 5, plus M0+1, then see if partial remainder steps suffice.
        M0 = health // 5
        
        best_L = None
        for M in [M0, M0 + 1]:
            if M < 0:
                continue
            base = 5 * M
            # Try r in 0..2, pick the first that achieves enough damage
            for r in range(3):
                dmg = base + partial_damage[offset][r]
                if dmg >= health:
                    L = 3 * M + r
                    if best_L is None or L < best_L:
                        best_L = L
                    break
            if best_L is not None:
                break
        
        T += best_L
    
    print(T)

# Do not forget to call main()
if __name__ == "__main__":
    main()