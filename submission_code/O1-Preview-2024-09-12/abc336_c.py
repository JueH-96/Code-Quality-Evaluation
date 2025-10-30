# YOUR CODE HERE
import sys

def main():
    import sys
    sys.setrecursionlimit(100000)
    N = int(sys.stdin.readline())
    D = ['0', '2', '4', '6', '8']
    D_nonzero = ['2','4','6','8']
    if N ==1:
        print(0)
        return
    N_remaining = N -1  # Exclude '0' since we have already considered it
    L = 1
    while True:
        count_L = 4 * (5 ** (L -1))
        if N_remaining <= count_L:
            break
        N_remaining -= count_L
        L +=1
    # Now N_remaining is between 1 and count_L
    idx_first_digit = (N_remaining -1) // (5 ** (L -1))
    first_digit = D_nonzero[idx_first_digit]
    idx_rest = (N_remaining -1) % (5 ** (L -1))
    # Convert idx_rest to base 5 with length L-1
    digits = []
    for _ in range(L -1):
        digit = idx_rest %5
        digits.append(D[digit])
        idx_rest = idx_rest //5
    digits.reverse()
    result = first_digit + ''.join(digits)
    print(result)

if __name__ == '__main__':
    main()