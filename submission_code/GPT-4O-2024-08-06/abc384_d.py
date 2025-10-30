# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    S = int(data[1])
    A = list(map(int, data[2:2 + N]))
    
    # Extend the sequence to simulate the infinite sequence
    A_extended = A + A
    
    # Calculate prefix sums
    prefix_sums = [0] * (2 * N + 1)
    for i in range(1, 2 * N + 1):
        prefix_sums[i] = prefix_sums[i - 1] + A_extended[i - 1]
    
    # Two pointers to find a subarray with sum S
    left = 0
    for right in range(1, 2 * N + 1):
        while left < right and prefix_sums[right] - prefix_sums[left] > S:
            left += 1
        if prefix_sums[right] - prefix_sums[left] == S:
            print("Yes")
            return
    
    print("No")

if __name__ == "__main__":
    main()