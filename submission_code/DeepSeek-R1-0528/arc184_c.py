import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    total_levels = 100
    T = (1 << total_levels) - 1 - A[-1]
    
    candidates = set()
    candidates.add(1)
    candidates.add(T)
    
    for level in range(0, 6):
        length_segment = 1 << (total_levels - level)
        num_segments_this_level = 1 << level
        for s in range(0, min(num_segments_this_level, 10000)):
            mid = (2 * s + 1) * (length_segment // 2)
            for d in A:
                base_val = mid - d
                for dx in (-1, 0, 1):
                    candidate_i0 = base_val + dx
                    if 1 <= candidate_i0 <= T:
                        candidates.add(candidate_i0)
    
    min_g = 10**9
    for i0 in candidates:
        g_val = 0
        for d in A:
            x = i0 + d
            count_flips = 0
            total_length = 1 << total_levels
            for l in range(total_levels):
                mid_point = total_length // 2
                if x > mid_point:
                    count_flips += 1
                    x = total_length - x
                elif x == mid_point:
                    break
                total_length = mid_point
                if total_length == 0:
                    break
            g_val += count_flips & 1
        if g_val < min_g:
            min_g = g_val
    answer = len(A) - min_g
    print(answer)

if __name__ == "__main__":
    main()