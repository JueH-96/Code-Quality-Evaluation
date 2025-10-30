def count_trailing_zeros(x):
    count = 0
    while (x & 1) == 0 and x != 0:
        count += 1
        x >>= 1
    return count

def main():
    import sys
    L, R = map(int, sys.stdin.readline().split())
    segments = []
    current_end = R
    while current_end > L:
        max_m = count_trailing_zeros(current_end)
        found = False
        for m in range(max_m, -1, -1):
            step = 1 << m
            if current_end - step >= L:
                start = current_end - step
                segments.append((start, current_end))
                current_end = start
                found = True
                break
    segments.reverse()
    print(len(segments))
    for seg in segments:
        print(seg[0], seg[1])

if __name__ == "__main__":
    main()