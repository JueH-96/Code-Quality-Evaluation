# YOUR CODE HERE
def max_strawberries(N, K, S):
    count = 0
    i = 0
    
    while i <= N - K:
        if S[i:i+K] == 'O' * K:
            count += 1
            i += K  # Skip the next K teeth as they become unhealthy
        else:
            i += 1
    
    return count

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    K = int(data[1])
    S = data[2]
    print(max_strawberries(N, K, S))