def main():
    import sys
    L, R = map(int, sys.stdin.readline().split())
    current_start = L
    result = []
    while current_start < R:
        s = R - current_start
        m_max = s.bit_length() - 1
        if current_start == 0:
            e = m_max
        else:
            # Compute the exponent of 2 in current_start
            e = (current_start & -current_start).bit_length() - 1
        k = min(e, m_max)
        step = 1 << k
        next_start = current_start + step
        result.append((current_start, next_start))
        current_start = next_start
    print(len(result))
    for l, r in result:
        print(l, r)

if __name__ == "__main__":
    main()