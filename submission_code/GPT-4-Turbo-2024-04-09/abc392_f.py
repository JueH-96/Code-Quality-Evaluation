import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    P = list(map(int, data[1:]))
    
    A = []
    for i in range(N):
        index = P[i] - 1
        A.insert(index, i + 1)
    
    print(" ".join(map(str, A)))

if __name__ == "__main__":
    main()