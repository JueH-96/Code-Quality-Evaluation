def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    
    # Parse the first two values: N and K
    N = int(data[0])
    K = int(data[1])
    
    # Compute the total sum of the numbers from 1 to K.
    total_sum = K * (K + 1) // 2
    
    # Create a set for the distinct numbers from A that fall within [1, K].
    present_numbers = set()
    for i in range(2, 2 + N):
        num = int(data[i])
        if 1 <= num <= K:
            present_numbers.add(num)
    
    # The result is the total sum minus the sum of numbers that are present.
    result = total_sum - sum(present_numbers)
    
    sys.stdout.write(str(result))

if __name__ == '__main__':
    main()