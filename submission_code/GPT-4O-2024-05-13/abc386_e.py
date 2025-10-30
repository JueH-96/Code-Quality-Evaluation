# YOUR CODE HERE
import sys
input = sys.stdin.read

def max_xor_subset(N, K, A):
    # Sort the array in descending order
    A.sort(reverse=True)
    
    # Initialize the result
    result = 0
    
    # Iterate over each bit position from the highest to the lowest
    for bit in range(59, -1, -1):
        # Try to maximize the result by setting the current bit
        temp_result = result | (1 << bit)
        
        # Count how many numbers can contribute to this bit being set
        count = 0
        for num in A:
            if (num & temp_result) == temp_result:
                count += 1
        
        # If we have at least K numbers that can contribute to this bit being set
        if count >= K:
            result = temp_result
    
    return result

def main():
    data = input().strip().split()
    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:]))
    
    print(max_xor_subset(N, K, A))

if __name__ == "__main__":
    main()