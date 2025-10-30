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
        queries.append((L-1, R))  # converting to 0-based index for L
    
    # Precompute prefix sums for '1's and '2's
    prefix_1 = [0] * (N + 1)
    prefix_2 = [0] * (N + 1)
    prefix_slash = [0] * (N + 1)
    
    for i in range(N):
        prefix_1[i+1] = prefix_1[i] + (1 if S[i] == '1' else 0)
        prefix_2[i+1] = prefix_2[i] + (1 if S[i] == '2' else 0)
        prefix_slash[i+1] = prefix_slash[i] + (1 if S[i] == '/' else 0)
    
    for query in queries:
        L, R = query
        # Count the number of '1's, '2's, and '/' in the substring
        count_1 = prefix_1[R] - prefix_1[L]
        count_2 = prefix_2[R] - prefix_2[L]
        count_slash = prefix_slash[R] - prefix_slash[L]
        
        # The maximum possible length is min(count_1, count_2) * 2 + 1 if there is at least one '/'
        # But since the '/' must be in the middle, we need to ensure that the '/' is present
        if count_slash == 0:
            print(0)
            continue
        
        # The length of the 11/22 string is 2 * min(count_1, count_2) + 1
        # But we need to ensure that the '/' is in the correct position
        # Since the '/' must be in the middle, the total length must be odd
        # So the maximum possible length is min(2 * min(count_1, count_2) + 1, R - L + 1)
        # But since the '/' must be in the middle, the length must be odd
        # So the maximum possible length is min(2 * min(count_1, count_2) + 1, R - L + 1) and must be odd
        
        # The length of the 11/22 string is 2 * k + 1, where k is the number of '1's and '2's on each side
        # So k = min(count_1, count_2)
        k = min(count_1, count_2)
        max_len = 2 * k + 1
        
        # Ensure that the length does not exceed the substring length
        max_len = min(max_len, R - L)
        
        # Ensure that the length is odd
        if max_len % 2 == 0:
            max_len -= 1
        
        # Ensure that there is at least one '/' in the substring
        if max_len < 1:
            print(0)
        else:
            print(max_len)

if __name__ == "__main__":
    main()