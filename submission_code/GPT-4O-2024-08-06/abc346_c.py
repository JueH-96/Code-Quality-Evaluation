# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:]))
    
    # Calculate the sum of numbers from 1 to K
    sum_1_to_K = K * (K + 1) // 2
    
    # Filter A to only include numbers <= K and remove duplicates
    relevant_A = set(x for x in A if x <= K)
    
    # Calculate the sum of relevant elements in A
    sum_relevant_A = sum(relevant_A)
    
    # The result is the difference
    result = sum_1_to_K - sum_relevant_A
    
    print(result)