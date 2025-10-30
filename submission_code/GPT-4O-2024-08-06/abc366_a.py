# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().strip()
    N, T, A = map(int, data.split())
    
    # Calculate the majority threshold
    majority_threshold = N // 2
    
    # Determine if the outcome is already decided
    if T > majority_threshold or A > majority_threshold:
        print("Yes")
    else:
        print("No")