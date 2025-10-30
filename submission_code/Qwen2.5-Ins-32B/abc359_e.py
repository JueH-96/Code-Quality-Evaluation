import sys

def solve():
    N = int(input())
    H = list(map(int, input().split()))
    
    # Initialize the answer array
    ans = [0] * N
    
    # Initialize the cumulative sum array
    cum_sum = [0] * (N + 1)
    
    # Iterate over the sequence H
    for i in range(N):
        # Update the cumulative sum
        cum_sum[i + 1] = cum_sum[i] + H[i]
        
        # Calculate the answer for the current position
        ans[i] = cum_sum[i] + 1
    
    # Adjust the answers based on the cumulative sum
    for i in range(N - 1, 0, -1):
        ans[i] = max(ans[i], ans[i - 1] + H[i])
    
    # Print the answers
    print(' '.join(map(str, ans)))

if __name__ == "__main__":
    solve()