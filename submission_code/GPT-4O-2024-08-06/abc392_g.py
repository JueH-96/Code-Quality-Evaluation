# YOUR CODE HERE
def count_fine_triplets():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    S = list(map(int, data[1:]))
    
    S_set = set(S)
    count = 0
    
    for i in range(N):
        A = S[i]
        for j in range(i + 1, N):
            B = S[j]
            C = 2 * B - A
            if C in S_set and C > B:
                count += 1
    
    print(count)