import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    
    nxt = [0] * (n + 1)
    front = None
    for i in range(n):
        x = A[i]
        if x == -1:
            front = i + 1
        else:
            nxt[x] = i + 1
            
    line = []
    current = front
    while current != 0:
        line.append(str(current))
        current = nxt[current]
        
    print(" ".join(line))

if __name__ == "__main__":
    main()