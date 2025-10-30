# YOUR CODE HERE
def count_inversions(arr, max_value):
    # Fenwick Tree (Binary Indexed Tree) for counting inversions
    fenwick_tree = [0] * (max_value + 1)
    
    def add(index, value):
        while index <= max_value:
            fenwick_tree[index] += value
            index += index & -index
    
    def sum(index):
        total = 0
        while index > 0:
            total += fenwick_tree[index]
            index -= index & -index
        return total
    
    inversions = 0
    for i in range(len(arr) - 1, -1, -1):
        # Count how many numbers are less than the current number
        inversions += sum(arr[i])
        # Add the current number to the Fenwick Tree
        add(arr[i] + 1, 1)
    
    return inversions

def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:2+N]))
    
    results = []
    
    for k in range(M):
        B = [(A[i] + k) % M for i in range(N)]
        # Since B[i] is in range [0, M-1], we can use M as the max_value for Fenwick Tree
        inversion_count = count_inversions(B, M)
        results.append(inversion_count)
    
    for result in results:
        print(result)