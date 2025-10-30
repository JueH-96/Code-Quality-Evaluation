def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    n = int(input_data[0])
    total_A = 0
    max_diff = -10**18  # Using a very small number for initialization
    index = 1
    for _ in range(n):
        A = int(input_data[index])
        B = int(input_data[index + 1])
        index += 2
        total_A += A
        # Calculate the extra height for being the top giant
        diff = B - A
        if diff > max_diff:
            max_diff = diff
    # The best stacking order is achieved by putting the giant with the maximum (B - A) on top.
    # Final height = (sum of all A's) + (max(B - A))
    answer = total_A + max_diff
    sys.stdout.write(str(answer))
    
if __name__ == '__main__':
    main()