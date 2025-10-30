# YOUR CODE HERE
import sys

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    Q = int(data[1])
    S = data[2]
    queries = []
    for i in range(Q):
        L = int(data[3 + 2*i])
        R = int(data[4 + 2*i])
        queries.append((L-1, R))  # converting to 0-based index
    
    # Precompute prefix sums for '1', '2', and '/'
    prefix_1 = [0] * (N + 1)
    prefix_2 = [0] * (N + 1)
    prefix_slash = [0] * (N + 1)
    
    for i in range(N):
        prefix_1[i+1] = prefix_1[i] + (1 if S[i] == '1' else 0)
        prefix_2[i+1] = prefix_2[i] + (1 if S[i] == '2' else 0)
        prefix_slash[i+1] = prefix_slash[i] + (1 if S[i] == '/' else 0)
    
    for query in queries:
        L, R = query
        count_1 = prefix_1[R] - prefix_1[L]
        count_2 = prefix_2[R] - prefix_2[L]
        count_slash = prefix_slash[R] - prefix_slash[L]
        
        if count_slash == 0:
            print(0)
            continue
        
        # The maximum possible length is 2 * min(count_1, count_2) + 1
        # But we need to ensure that there is at least one '/'
        # So the length is min(2 * min(count_1, count_2) + 1, count_1 + count_2 + 1)
        # But since the '/' must be in the middle, we need to have at least one '/' in the substring
        # So the maximum length is 2 * min(count_1, count_2) + 1, but only if there is at least one '/'
        max_len = 2 * min(count_1, count_2) + 1
        if max_len < 1:
            print(0)
        else:
            print(max_len)

if __name__ == "__main__":
    main()