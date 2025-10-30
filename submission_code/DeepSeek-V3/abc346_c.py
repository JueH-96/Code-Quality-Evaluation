# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:2+N]))
    
    # Create a set of unique elements in A that are <= K
    unique_A = set()
    for num in A:
        if num <= K:
            unique_A.add(num)
    
    # Calculate the sum of numbers from 1 to K
    total_sum = K * (K + 1) // 2
    
    # Calculate the sum of numbers in A that are <= K
    sum_A = sum(unique_A)
    
    # The result is the total sum minus the sum of numbers in A that are <= K
    result = total_sum - sum_A
    
    print(result)

if __name__ == "__main__":
    main()