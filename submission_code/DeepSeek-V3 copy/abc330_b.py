# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    L = int(data[1])
    R = int(data[2])
    A = list(map(int, data[3:3+N]))
    
    X = []
    for a in A:
        if a < L:
            X.append(L)
        elif a > R:
            X.append(R)
        else:
            X.append(a)
    
    print(' '.join(map(str, X)))

if __name__ == "__main__":
    main()