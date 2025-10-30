# YOUR CODE HERE
def alternating_zeros_ones(N):
    result = []
    for i in range(N):
        result.append('10')
    result.append('1')
    print(''.join(result))

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    N = int(input().strip())
    alternating_zeros_ones(N)