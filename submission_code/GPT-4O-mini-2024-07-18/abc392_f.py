def insert_elements(N, P):
    A = []
    for i in range(1, N + 1):
        # P[i-1] is the position where we want to insert i
        position = P[i - 1] - 1  # Convert to 0-based index
        A.insert(position, i)
    return A

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    P = list(map(int, data[1:N + 1]))
    
    result = insert_elements(N, P)
    print(" ".join(map(str, result)))