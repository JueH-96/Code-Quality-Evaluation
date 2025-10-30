# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    Q = int(data[1])
    S = data[2]
    
    queries = []
    index = 3
    for _ in range(Q):
        L = int(data[index])
        R = int(data[index + 1])
        queries.append((L, R))
        index += 2
    
    # Precompute prefix sums for '1', '2', and '/'
    prefix_1 = [0] * (N + 1)
    prefix_2 = [0] * (N + 1)
    prefix_slash = [0] * (N + 1)
    
    for i in range(1, N + 1):
        prefix_1[i] = prefix_1[i - 1] + (1 if S[i - 1] == '1' else 0)
        prefix_2[i] = prefix_2[i - 1] + (1 if S[i - 1] == '2' else 0)
        prefix_slash[i] = prefix_slash[i - 1] + (1 if S[i - 1] == '/' else 0)
    
    results = []
    
    for L, R in queries:
        count_1 = prefix_1[R] - prefix_1[L - 1]
        count_2 = prefix_2[R] - prefix_2[L - 1]
        count_slash = prefix_slash[R] - prefix_slash[L - 1]
        
        # We need at least one '/' to form an 11/22 string
        if count_slash == 0:
            results.append(0)
            continue
        
        # The maximum length of 11/22 string we can form
        max_length = 2 * min(count_1, count_2) + 1
        results.append(max_length)
    
    for result in results:
        print(result)