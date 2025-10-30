# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    B = A[:]  # Start with the initial stones
    total_giftable_stones = 0
    
    for i in range(N):
        # The i-th alien becomes an adult
        B[i] += total_giftable_stones
        
        # Update total_giftable_stones for future adults
        if B[i] > 0:
            total_giftable_stones += 1
    
    print(' '.join(map(str, B)))