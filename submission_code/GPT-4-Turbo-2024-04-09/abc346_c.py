def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:2+N]))
    
    # Convert list A to a set for fast lookup
    set_A = set(A)
    
    # Calculate the sum of all numbers from 1 to K
    total_sum = K * (K + 1) // 2
    
    # Subtract the sum of numbers that are in A and also <= K
    sum_in_A = sum(x for x in set_A if x <= K)
    
    # The result is the total sum minus the sum of numbers in A that are <= K
    result = total_sum - sum_in_A
    
    print(result)

if __name__ == "__main__":
    main()