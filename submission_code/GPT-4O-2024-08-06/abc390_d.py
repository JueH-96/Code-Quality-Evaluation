# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    # Set to store all unique XOR results
    xor_results = set()
    
    # There are 2^N possible subsets
    for subset_mask in range(1 << N):
        current_xor = 0
        for i in range(N):
            if subset_mask & (1 << i):
                current_xor ^= A[i]
        xor_results.add(current_xor)
    
    # The number of unique XOR results is the answer
    print(len(xor_results))