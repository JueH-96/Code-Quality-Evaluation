# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    total_score = 0
    for i in range(N - 1):
        total_score += abs(A[i] - A[i + 1])
    
    print(total_score)