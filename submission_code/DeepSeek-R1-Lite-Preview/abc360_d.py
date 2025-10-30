import sys
import bisect

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    T = int(data[1])
    S = data[2]
    X = list(map(int, data[3:3+N]))
    
    # Pair positions with directions and sort by position
    ants = sorted(zip(X, S), key=lambda x: x[0])
    positions = [ant[0] for ant in ants]
    directions = [ant[1] for ant in ants]
    
    # Separate into right-moving and left-moving ants
    R = []
    L = []
    for pos, dir in zip(positions, directions):
        if dir == '1':
            R.append(pos)
        else:
            L.append(pos)
    
    threshold = 2 * (T + 0.1)
    count = 0
    
    # For each right-moving ant, find left-moving ants within the threshold
    for ri in R:
        lower = bisect.bisect_right(L, ri)
        upper = bisect.bisect_left(L, ri + threshold)
        count += upper - lower
    
    print(count)

if __name__ == '__main__':
    main()