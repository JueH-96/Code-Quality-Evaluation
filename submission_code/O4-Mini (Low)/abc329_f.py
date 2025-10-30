import sys
import threading

def main():
    import sys
    input = sys.stdin.readline
    
    N, Q = map(int, input().split())
    colors = list(map(int, input().split()))
    
    # For each box i, we keep a dictionary maps[i] that maps color -> count.
    maps = [dict() for _ in range(N + 1)]
    for i, c in enumerate(colors, start=1):
        maps[i][c] = 1
    
    out = []
    for _ in range(Q):
        a, b = map(int, input().split())
        # If a == b, nothing really moves, but problem guarantees a != b.
        # To optimize merging, always merge the smaller dictionary into the larger one.
        if len(maps[a]) > len(maps[b]):
            # swap so that maps[a] is the smaller
            maps[a], maps[b] = maps[b], maps[a]
        
        # Now maps[a] is smaller or equal. Merge it into maps[b].
        small = maps[a]
        large = maps[b]
        for color, cnt in small.items():
            large[color] = large.get(color, 0) + cnt
        
        # Clear the source box a
        small.clear()
        
        # The answer is the number of distinct colors in box b
        out.append(str(len(large)))
    
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()