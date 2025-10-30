def count_trailing_zeros(N):
    count = 0
    while (N & 1) == 0:  # Check if the rightmost bit is 0
        count += 1
        N >>= 1  # Right shift to check the next bit
    
    return count

N = int(input())
print(count_trailing_zeros(N))