import sys
import threading

def main():
    data = sys.stdin.read().strip().split()
    n = int(data[0])
    H = list(map(int, data[1:]))

    T = 0
    for h in H:
        # Number of full 3-attacks cycles (each cycle of 3 attacks does total damage 5)
        full_cycles = h // 5
        rem = h - full_cycles * 5
        k = full_cycles * 3  # attacks used in full cycles
        
        if rem > 0:
            # We need a few more attacks (at most 3) to cover 'rem' damage.
            # The damage pattern depends on T % 3.
            s0 = T % 3
            acc = 0
            # attack 1
            d = 3 if (s0 + 1) % 3 == 0 else 1
            acc += d; k += 1
            if acc < rem:
                # attack 2
                d = 3 if (s0 + 2) % 3 == 0 else 1
                acc += d; k += 1
                if acc < rem:
                    # attack 3
                    d = 3 if (s0 + 3) % 3 == 0 else 1  # same as (s0)%3 == 0
                    acc += d; k += 1
                    # Now acc must be >= rem, since 1+1+3 = 5 covers up to rem=4
        T += k

    print(T)

if __name__ == "__main__":
    main()