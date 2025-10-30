import sys

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    last_positions = {}
    B = []
    for i in range(N):
        if A[i] in last_positions:
            B.append(str(last_positions[A[i]]))
        else:
            B.append(str(-1))
        last_positions[A[i]] = i + 1  # positions are 1-based
    print(' '.join(B))

if __name__ == "__main__":
    main()