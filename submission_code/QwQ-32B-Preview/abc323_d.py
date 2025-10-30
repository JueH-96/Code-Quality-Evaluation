import sys
from collections import defaultdict

def main():
    import math
    N = int(sys.stdin.readline())
    slimes = defaultdict(int)
    sizes = []
    for _ in range(N):
        S, C = map(int, sys.stdin.readline().split())
        slimes[S] += C
        sizes.append(S)
    
    sizes.sort()
    max_size = sizes[-1] * 2  # To handle the largest possible synthesized size
    processed = set()
    
    for X in sizes:
        if X in processed:
            continue
        while X <= max_size:
            if X in slimes:
                C_X = slimes[X]
                unpaired = C_X % 2
                paired = C_X // 2
                if paired > 0:
                    slimes[2 * X] += paired
                slimes[X] = unpaired
                X *= 2
            else:
                break
        processed.add(X)
    
    min_slimes = sum(slimes[X] for X in slimes)
    print(min_slimes)

if __name__ == "__main__":
    main()