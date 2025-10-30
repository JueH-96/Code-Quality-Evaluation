# YOUR CODE HERE
def main():
    N,*rest = map(int, open(0).read().split())
    A = rest
    pos = {}
    for idx, color in enumerate(A,1):
        if color not in pos:
            pos[color] = []
        pos[color].append(idx)
    count = 0
    for color in range(1, N+1):
        p = pos[color]
        if len(p) ==2 and abs(p[1]-p[0]) ==2:
            count +=1
    print(count)

if __name__ == "__main__":
    main()