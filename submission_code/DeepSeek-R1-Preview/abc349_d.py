def main():
    import sys
    L, R = map(int, sys.stdin.readline().split())
    current = L
    result = []
    
    while current < R:
        if current == 0:
            m = R
            s_max = (m).bit_length() - 1
            step = 1 << s_max
        else:
            x = current
            t = x & -x
            v = (t).bit_length() - 1
            m = R - current
            if m == 0:
                break
            s_max = (m).bit_length() - 1
            s = min(v, s_max)
            step = 1 << s
        result.append((current, current + step))
        current += step
    
    print(len(result))
    for l, r in result:
        print(l, r)

if __name__ == "__main__":
    main()