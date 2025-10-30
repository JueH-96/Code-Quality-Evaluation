# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    total_sum = 0
    
    for i in range(N - 1):
        current_xor = 0
        for j in range(i + 1, N):
            current_xor ^= A[j]
            total_sum += current_xor
    
    print(total_sum)