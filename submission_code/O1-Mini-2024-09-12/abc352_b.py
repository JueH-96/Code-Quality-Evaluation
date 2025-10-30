import sys
import bisect

def main():
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()
    
    from collections import defaultdict
    char_positions = defaultdict(list)
    for idx, c in enumerate(T):
        char_positions[c].append(idx + 1)
    
    result = []
    current = 0
    for c in S:
        positions = char_positions[c]
        i = bisect.bisect_right(positions, current)
        if i < len(positions):
            pos = positions[i]
            result.append(str(pos))
            current = pos
        else:
            # According to problem constraints, this should not happen
            pass
    
    print(' '.join(result))

if __name__ == "__main__":
    main()