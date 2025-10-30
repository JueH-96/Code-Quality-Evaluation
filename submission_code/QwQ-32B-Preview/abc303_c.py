import sys

def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    M = int(data[1])
    H = int(data[2])
    K = int(data[3])
    S = data[4]
    items = set()
    for i in range(M):
        x = int(data[5 + 2 * i])
        y = int(data[6 + 2 * i])
        items.add((x, y))
    
    x, y = 0, 0
    for move in S:
        if move == 'R':
            x += 1
        elif move == 'L':
            x -= 1
        elif move == 'U':
            y += 1
        else:  # 'D'
            y -= 1
        H -= 1
        if H < 0:
            print('No')
            return
        if (x, y) in items and H < K:
            H = K
    print('Yes')

if __name__ == '__main__':
    main()