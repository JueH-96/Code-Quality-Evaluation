# YOUR CODE HERE
import sys

def main():
    import sys
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    H = int(data[2])
    K = int(data[3])
    S = data[4]
    items = set()
    for i in range(5, 5 + 2*M, 2):
        x = int(data[i])
        y = int(data[i+1])
        items.add( (x, y) )
    x, y = 0, 0
    for move in S:
        if move == 'R':
            x +=1
        elif move == 'L':
            x -=1
        elif move == 'U':
            y +=1
        elif move == 'D':
            y -=1
        H -=1
        if H <0:
            print("No")
            return
        if (x, y) in items and H < K:
            H = K
            items.remove( (x, y) )
    print("Yes")

if __name__ == "__main__":
    main()