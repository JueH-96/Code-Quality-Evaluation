def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    M = int(data[1])
    S = data[2]
    
    # Initially, Takahashi has M plain T-shirts.
    available_plain = M
    available_logo = 0  # Initially, he has no logo T-shirts.
    used_plain = 0      # T-shirts used since the last washing.
    used_logo = 0
    purchased = 0       # Counts logo T-shirts purchased along the way.
    
    # Process the schedule day by day.
    for ch in S:
        if ch == '0':
            # On days with no plans, he doesn't wear a T-shirt and washes all used ones.
            available_plain += used_plain
            available_logo += used_logo
            used_plain = 0
            used_logo = 0
        elif ch == '1':
            # On meal days, he can wear either a plain or a logo T-shirt.
            # Strategy: Use a plain T-shirt first (if available) to save logo ones for competitive events.
            if available_plain > 0:
                available_plain -= 1
                used_plain += 1
            elif available_logo > 0:
                available_logo -= 1
                used_logo += 1
            else:
                # No available T-shirts; he must purchase a new logo T-shirt.
                purchased += 1
                used_logo += 1
        elif ch == '2':
            # On competitive programming event days, he must wear a logo T-shirt.
            if available_logo > 0:
                available_logo -= 1
                used_logo += 1
            else:
                purchased += 1
                used_logo += 1

    # Output the number of logo T-shirts he needed to buy.
    sys.stdout.write(str(purchased))

if __name__ == '__main__':
    main()