# YOUR CODE HERE
def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    A = list(map(int, data[1:1+N]))
    B = list(map(int, data[1+N:1+2*N]))
    print(max(A) + max(B))

main()