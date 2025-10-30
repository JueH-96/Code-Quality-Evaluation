# YOUR CODE HERE
import sys

def main():
    """
    Reads input, solves the problem, and prints the output.
    """
    # Use fast I/O
    readline = sys.stdin.readline
    
    # Read problem inputs
    try:
        N = int(readline())
        # Handle empty input case
        if N == 0:
            return
        H = list(map(int, readline().split()))
    except (ValueError, IndexError):
        return

    # Step 1: Compute the 'previous greater element' array L.
    # L[j] will store the index of the nearest building to the left of j that is taller than H[j].
    # If no such building exists, L[j] remains -1.
    L = [-1] * N
    stack = []  # Stack stores indices, maintaining H[stack elements] in decreasing order.
    for j in range(N):
        while stack and H[stack[-1]] <= H[j]:
            stack.pop()
        if stack:
            L[j] = stack[-1]
        stack.append(j)
        
    # Step 2: Use a difference array for efficient range updates.
    # For a fixed j, it contributes 1 to the count for all i in the range [max(0, L[j]), j-1].
    # We apply this by incrementing at the start of the range and decrementing after the end.
    diff = [0] * (N + 1)
    for j in range(1, N):
        p = L[j]
        
        # Determine the start of the range of i's that get a +1 from this j.
        # If L[j] = p >= 0, the range is [p, j-1].
        # If L[j] = -1, the range is [0, j-1].
        # This is equivalent to starting at max(0, p).
        start_idx = max(0, p)
        end_idx = j - 1
        
        if start_idx <= end_idx:
            diff[start_idx] += 1
            diff[end_idx + 1] -= 1

    # Step 3: Calculate the final answer by taking the prefix sum of the difference array.
    # The value at each index of the final answer array is the cumulative sum of the diff array.
    ans = [0] * N
    current_sum = 0
    for i in range(N):
        current_sum += diff[i]
        ans[i] = current_sum
        
    # Print the results space-separated.
    print(*ans)

if __name__ == "__main__":
    main()