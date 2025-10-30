# YOUR CODE HERE
import sys
input = sys.stdin.read

def count_black_squares(N, P, queries):
    # Precompute the number of black squares in the NxN pattern
    black_count = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(N):
        for j in range(N):
            black_count[i + 1][j + 1] = black_count[i + 1][j] + black_count[i][j + 1] - black_count[i][j] + (1 if P[i][j] == 'B' else 0)
    
    results = []
    for A, B, C, D in queries:
        # Calculate the number of full NxN blocks in the area
        full_blocks = ((C - A + 1) // N) * ((D - B + 1) // N)
        # Calculate the number of black squares in these full blocks
        total_black = full_blocks * black_count[N][N]
        
        # Calculate the remaining areas
        remaining_rows = (C - A + 1) % N
        remaining_cols = (D - B + 1) % N
        
        # Calculate the number of black squares in the remaining areas
        for i in range(remaining_rows):
            for j in range(remaining_cols):
                if P[(A + i) % N][(B + j) % N] == 'B':
                    total_black += 1
        
        # Calculate the number of black squares in the remaining rows
        for i in range(remaining_rows):
            for j in range((D - B + 1) // N):
                total_black += black_count[i + 1][N] - black_count[i][N]
        
        # Calculate the number of black squares in the remaining columns
        for i in range((C - A + 1) // N):
            for j in range(remaining_cols):
                total_black += black_count[N][j + 1] - black_count[N][j]
        
        results.append(total_black)
    
    return results

def main():
    data = input().split()
    N = int(data[0])
    Q = int(data[1])
    P = []
    index = 2
    for _ in range(N):
        P.append(data[index])
        index += 1
    queries = []
    for _ in range(Q):
        A = int(data[index])
        B = int(data[index + 1])
        C = int(data[index + 2])
        D = int(data[index + 3])
        queries.append((A, B, C, D))
        index += 4
    
    results = count_black_squares(N, P, queries)
    for result in results:
        print(result)

if __name__ == "__main__":
    main()