def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    M = int(data[1])
    S = data[2].strip()
    
    # Given X extra logo T-shirts purchased, we simulate the N days.
    # Initially, all T-shirts (plain and purchased logo) are washed and ready.
    # On days:
    #  "0": no T-shirt is worn and all used T-shirts get washed.
    #  "1": wear a T-shirt for meal; prefer plain (if available) to save logo T-shirts.
    #  "2": must wear a logo T-shirt.
    # Once a T-shirt is worn in a segment (between wash days), it cannot be used again.
    def simulate(X):
        plain = M  # available plain T-shirts for the current segment
        logo = X   # available logo T-shirts for the current segment (purchased ones)
        for ch in S:
            if ch == '0':
                # Wash day: all worn T-shirts become available again.
                plain = M
                logo = X
            elif ch == '1':
                # Meal day: use a plain shirt if possible; otherwise, use a logo shirt.
                if plain > 0:
                    plain -= 1
                elif logo > 0:
                    logo -= 1
                else:
                    return False
            elif ch == '2':
                # Competitive programming day: must use a logo shirt.
                if logo > 0:
                    logo -= 1
                else:
                    return False
        return True

    # Since N is at most 1000, we try possible X values from 0 up to N.
    for X in range(0, N+1):
        if simulate(X):
            sys.stdout.write(str(X))
            return

if __name__ == '__main__':
    main()