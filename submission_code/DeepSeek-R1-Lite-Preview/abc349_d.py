def main():
    import sys
    L_str, R_str = sys.stdin.read().split()
    L = int(L_str)
    R = int(R_str)
    
    divisions = []
    current = L
    while current < R:
        if current == 0:
            # For current = 0, choose the largest possible i where 0 + 2^i <= R
            i = 0
            while (1 << i) <= (R - current):
                i += 1
            i -= 1
        else:
            # Find the number of trailing zeros in current
            k = (current & -current).bit_length() - 1
            # Try i from k down to 0
            for i in range(k, -1, -1):
                if current + (1 << i) <= R:
                    break
        divisions.append((current, current + (1 << i)))
        current += (1 << i)
    
    print(len(divisions))
    for l, r in divisions:
        print(l, r)

if __name__ == "__main__":
    main()