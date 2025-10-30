# YOUR CODE HERE
import sys
input = sys.stdin.read

def min_operations_to_balance(N, A):
    total_sum = sum(A)
    avg = total_sum // N
    remainder = total_sum % N
    
    # We need exactly 'remainder' elements to be 'avg + 1'
    # and the rest to be 'avg'
    # The number of operations required to achieve this is:
    # sum(max(0, A_i - (avg + 1)) + max(0, avg - A_i)) for all i
    
    operations = 0
    for a in A:
        if a > avg + 1:
            operations += a - (avg + 1)
        elif a < avg:
            operations += avg - a
    
    return operations

def main():
    data = input().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))
    print(min_operations_to_balance(N, A))

if __name__ == "__main__":
    main()