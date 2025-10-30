import sys

def main():
    N = int(sys.stdin.readline())
    
    total_A_sum = 0
    # Since B_i >= A_i by problem constraint, B_i - A_i >= 0.
    # So, initializing max_B_minus_A to 0 is correct.
    # Any valid B_i - A_i will be >= 0. If all are 0, max_B_minus_A remains 0.
    max_B_minus_A = 0 
    
    for _ in range(N):
        A, B = map(int, sys.stdin.readline().split())
        
        total_A_sum += A
        
        diff = B - A
        if diff > max_B_minus_A:
            max_B_minus_A = diff
            
    result = total_A_sum + max_B_minus_A
    print(result)

if __name__ == '__main__':
    main()