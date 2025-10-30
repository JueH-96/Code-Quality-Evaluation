def max_total_score(N, A):
    # Calculate the total score by summing the absolute differences of adjacent elements
    total_score = 0
    for i in range(N - 1):
        total_score += abs(A[i] - A[i + 1])
    
    return total_score

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N + 1]))
    
    result = max_total_score(N, A)
    print(result)