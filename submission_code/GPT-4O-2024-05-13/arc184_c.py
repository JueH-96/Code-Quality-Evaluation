# YOUR CODE HERE
def is_mountain_fold(index):
    # Count the number of 1s in the binary representation of index
    # If the number of 1s is odd, it's a mountain fold
    # If the number of 1s is even, it's a valley fold
    return bin(index).count('1') % 2 == 1

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    max_value = 0
    
    for i in range(1, 2**100 - A[-1]):
        count = 0
        for k in range(N):
            if is_mountain_fold(i + A[k]):
                count += 1
        max_value = max(max_value, count)
    
    print(max_value)

if __name__ == "__main__":
    main()