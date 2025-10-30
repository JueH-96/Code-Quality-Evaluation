# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    T = int(data[1])
    A = list(map(int, data[2:]))
    
    rows = [0] * N
    cols = [0] * N
    diag1 = 0
    diag2 = 0
    
    def get_position(num):
        return (num - 1) // N, (num - 1) % N
    
    for turn, num in enumerate(A, 1):
        r, c = get_position(num)
        rows[r] += 1
        cols[c] += 1
        if r == c:
            diag1 += 1
        if r + c == N - 1:
            diag2 += 1
        
        if any(x == N for x in [rows[r], cols[c], diag1, diag2]):
            print(turn)
            return
    
    print(-1)

if __name__ == "__main__":
    main()