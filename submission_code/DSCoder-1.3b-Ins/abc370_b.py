# YOUR CODE HERE

N = int(input())
A = [list(map(int, input().split())) for _ in range(N)]

def find_final_element(N, A):
    for i in range(N):
        for j in range(N):
            if i >= j:
                A[i][j] = A[j][i]
            else:
                A[i][j] = A[j][i]
    return A[0][0]

print(find_final_element(N, A))